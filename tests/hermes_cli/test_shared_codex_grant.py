"""Explicit grant references resolve through the real profile and auth-store seams."""
import base64
import json
from pathlib import Path
import time

import pytest

import hermes_constants
from hermes_cli import auth
from hermes_cli.auth_constants import DEFAULT_CODEX_BASE_URL
from hermes_cli.config import atomic_config_replace, atomic_config_write
from hermes_cli.runtime_provider import resolve_runtime_provider, resolve_runtime_with_fallback


PROVIDER = 'openai-codex'
ACCOUNT = 'fixture-account'
ROW = 'ab12cd'
SUBJECT = 'fixture-subject'


def token(account=ACCOUNT, *, subject=SUBJECT, expired=False, generation=1):
    claims = {'sub': subject, 'exp': int(time.time()) + (-60 if expired else 3600),
              'https://api.openai.com/auth': {'chatgpt_account_id': account}, 'generation': generation}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=')
    return f'e30.{payload}.fixture'


def row(row_id=ROW, account=ACCOUNT, **overrides):
    return {'id': row_id, 'source': 'manual:device_code', 'auth_type': 'oauth',
            'access_token': token(account), 'refresh_token': 'fixture-refresh',
            'base_url': DEFAULT_CODEX_BASE_URL, 'priority': 0, **overrides}


def config(**overrides):
    return {'auth': {'inherit_root_credentials': False, 'adopt_external_logins': False,
                     'shared_providers': {PROVIDER: {'source': 'root', 'credential_id': ROW, 'account_id': ACCOUNT,
                                                    'subject': SUBJECT}}},
            'model': {'provider': PROVIDER, 'default': 'gpt-6.2-sol'}, **overrides}


@pytest.fixture
def stores(tmp_path, monkeypatch):
    root = tmp_path / 'disposable-hermes-root'
    worker = root / 'profiles' / 'worker'
    worker.mkdir(parents=True)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(worker))
    monkeypatch.setattr(hermes_constants, '_default_hermes_root_memo', None)
    atomic_config_write(worker / 'config.yaml', config())
    return root, worker


def seed(root, *rows, singleton=None):
    state = {PROVIDER: {'tokens': singleton}} if singleton else {}
    auth._save_auth_store({'providers': state, 'active_provider': 'anthropic',
                           'credential_pool': {PROVIDER: list(rows), 'anthropic': [row()]}},
                          target_path=root / 'auth.json')


def test_explicit_named_provider_uses_only_referenced_root_row(stores, monkeypatch):
    root, worker = stores
    chosen = row()
    seed(root, row('111111', 'unrelated-account'), chosen)
    monkeypatch.setenv('HERMES_CODEX_BASE_URL', 'https://wrong.invalid')
    runtime = resolve_runtime_provider(requested=PROVIDER)
    assert runtime['api_key'] == chosen['access_token']
    assert runtime['base_url'] == DEFAULT_CODEX_BASE_URL
    assert runtime['source'] == 'shared-root-oauth'
    assert auth.read_credential_pool() == {}
    assert auth.read_credential_pool('anthropic') == []
    assert not (worker / 'auth.json').exists()


@pytest.mark.parametrize('replacement', [None, row(account='different-account'), row(last_status='dead'),
                                          row(access_token=token(subject='different-member')),
                                          row(auth_type='api_key'), row(source='env:FAKE')])
def test_missing_or_mismatched_grant_never_walks_another_provider(stores, replacement):
    root, _ = stores
    seed(root, row('111111', 'unrelated-account'), *([replacement] if replacement else []))
    cfg = config(fallback_providers=[{'provider': 'openrouter', 'model': 'never-used'}])
    with pytest.raises(auth.AuthError, match='Referenced root Codex OAuth grant'):
        resolve_runtime_with_fallback(cfg, requested=PROVIDER)


@pytest.mark.parametrize('shared', [None, {'anthropic': {'source': 'root'}},
    {PROVIDER: {'source': '../elsewhere', 'credential_id': ROW, 'account_id': ACCOUNT}},
    {PROVIDER: {'source': 'root', 'credential_id': '../escape', 'account_id': ACCOUNT}}])
def test_malformed_share_denies_root_before_store_lookup(stores, shared, monkeypatch):
    root, worker = stores
    cfg = config()
    cfg['auth']['shared_providers'] = shared
    cfg['auth']['inherit_root_credentials'] = True
    atomic_config_replace(worker / 'config.yaml', cfg)
    seed(root, row())
    tracked = []
    load = auth._load_auth_store

    def track(path=None, **kwargs):
        if path == root / 'auth.json':
            tracked.append(path)
        return load(path, **kwargs)

    monkeypatch.setattr(auth, '_load_auth_store', track)
    assert auth.read_credential_pool('anthropic') == []
    with pytest.raises(auth.AuthError, match='Invalid shared provider credential policy'):
        resolve_runtime_provider(requested=PROVIDER)
    assert not tracked


@pytest.mark.parametrize('rows', [(row(), row()), (row(source='device_code'),)])
def test_duplicate_id_or_orphaned_singleton_row_cannot_authorize(stores, rows):
    root, _ = stores
    seed(root, *rows)
    with pytest.raises(auth.AuthError, match='Referenced root Codex OAuth grant'):
        resolve_runtime_provider(requested=PROVIDER)


def test_environment_or_auto_does_not_opt_in_to_share(stores, monkeypatch):
    root, worker = stores
    seed(root, row())
    atomic_config_replace(worker / 'config.yaml', config(model={'default': 'gpt-6.2-sol'}))
    monkeypatch.setenv('HERMES_INFERENCE_PROVIDER', PROVIDER)
    with pytest.raises(auth.AuthError, match='explicit provider'):
        resolve_runtime_provider()
    with pytest.raises(auth.AuthError, match='explicit provider'):
        resolve_runtime_provider(requested='auto')
    assert resolve_runtime_provider(requested=PROVIDER)['source'] == 'shared-root-oauth'


@pytest.mark.parametrize('route', [{'base_url': 'https://proxy.invalid'}, {'api_mode': 'chat_completions'},
                                   {'openai_runtime': 'codex_app_server'}])
def test_share_refuses_non_native_route_before_reading_store(stores, route, monkeypatch):
    _, worker = stores
    atomic_config_replace(worker / 'config.yaml', config(model={'provider': PROVIDER, 'default': 'gpt-6.2-sol', **route}))
    def unexpected_io(*args, **kwargs):
        pytest.fail('an invalid route must not read root credentials')
    monkeypatch.setattr(auth, '_load_auth_store', unexpected_io)
    with pytest.raises(auth.AuthError, match='native provider endpoint and transport'):
        resolve_runtime_provider(requested=PROVIDER)


def test_shareable_list_is_root_only_read_only_and_metadata_only(stores, monkeypatch, capsys):
    from argparse import Namespace
    from hermes_cli.auth_commands import auth_list_command

    root, _ = stores
    selected = row(access_token=token(expired=True))
    seed(root, selected, row('external', source='codex_cli'), row('revoked', last_status='dead'))
    args = Namespace(provider=PROVIDER, shareable=True)
    with pytest.raises(SystemExit, match='main profile only'):
        auth_list_command(args)
    monkeypatch.setenv('HERMES_HOME', str(root))
    before = (root / 'auth.json').read_bytes()
    auth_list_command(args)
    assert capsys.readouterr().out == f'id={ROW} account_id={ACCOUNT} subject={SUBJECT}\n'
    assert (root / 'auth.json').read_bytes() == before


def test_primary_pool_with_stale_manual_grant_adopts_worker_rotation(stores, monkeypatch):
    from agent.credential_pool import load_pool

    root, worker = stores
    old = row()
    seed(root, old)
    monkeypatch.setenv('HERMES_HOME', str(root))
    primary_pool = load_pool(PROVIDER)
    initial = primary_pool.select()
    assert initial is not None and initial.access_token == old['access_token']
    expected = token(generation=2)
    calls = []

    def rotate(access, refresh, **kwargs):
        calls.append((access, refresh))
        assert refresh == old['refresh_token'], 'refreshing twice would consume the next generation'
        return {'access_token': expected, 'refresh_token': 'fixture-rotated', 'last_refresh': '2026-11-25T12:00:00Z'}

    monkeypatch.setattr(auth, 'refresh_codex_oauth_pure', rotate)
    monkeypatch.setenv('HERMES_HOME', str(worker))
    shared = auth.resolve_codex_runtime_credentials(force_refresh=True, stale_access_token=old['access_token'])
    assert shared['api_key'] == expected
    monkeypatch.setenv('HERMES_HOME', str(root))
    refreshed = primary_pool.try_refresh_matching(api_key_hint=old['access_token'])
    assert refreshed is not None and refreshed.access_token == expected
    assert calls == [(old['access_token'], old['refresh_token'])]
    assert not (worker / 'auth.json').exists()


def test_corrupt_named_config_cannot_freeze_its_last_good_reference(stores):
    from gateway.session_local import load_profile_config
    from hermes_cli.runtime_provider import frozen_runtime_config

    root, worker = stores
    seed(root, row())
    good = load_profile_config(worker)
    with frozen_runtime_config(good):
        assert resolve_runtime_provider(requested=PROVIDER)['source'] == 'shared-root-oauth'
    assert list((worker / 'backups' / 'config').glob('*.good.*'))
    (worker / 'config.yaml').write_text('auth: [broken\n')
    recovered = load_profile_config(worker)
    assert recovered['model'] == good['model'], 'non-auth settings can recover'
    with frozen_runtime_config(recovered), pytest.raises(auth.AuthError, match='Invalid shared provider credential policy'):
        resolve_runtime_provider(requested=PROVIDER)
    assert not (worker / 'auth.json').exists()


def test_frozen_worker_cannot_be_retargeted_by_later_config_edit(stores):
    from gateway.session_local import load_profile_config
    from hermes_cli.runtime_provider import frozen_runtime_config

    root, worker = stores
    selected = row()
    replacement = row('other', 'other-account')
    seed(root, selected, replacement)
    frozen = load_profile_config(worker)
    modified = config()
    modified['auth']['shared_providers'][PROVIDER].update(credential_id='other', account_id='other-account')
    atomic_config_replace(worker / 'config.yaml', modified)
    with frozen_runtime_config(frozen):
        assert resolve_runtime_provider(requested=PROVIDER)['api_key'] == selected['access_token']
    assert resolve_runtime_provider(requested=PROVIDER)['api_key'] == replacement['access_token']
