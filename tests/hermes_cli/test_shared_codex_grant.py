"""Explicit grant references resolve through the real profile and auth-store seams."""
import base64
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock, Thread
import time
from urllib.parse import parse_qs

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


@pytest.mark.parametrize('broken', ['auth: [broken\n', 'auth: {shared_providers: not-a-map}\n'])
def test_bad_shared_policy_preserves_nonshared_worker_credentials_only(stores, monkeypatch, broken):
    root, worker = stores
    (worker / 'config.yaml').write_text(broken)
    seed(root, row())
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-fixture-worker-only')
    load = auth._load_auth_store

    def read_own_only(path=None, **kwargs):
        assert path != root / 'auth.json', 'invalid config must never read root credentials'
        return load(path, **kwargs)

    monkeypatch.setattr(auth, '_load_auth_store', read_own_only)
    monkeypatch.setattr(auth, 'refresh_codex_oauth_pure', lambda *a, **k: pytest.fail('unexpected refresh'))
    for provider in ('anthropic', 'auto'):
        runtime = resolve_runtime_provider(requested=provider)
        assert (runtime['provider'], runtime['api_key']) == ('anthropic', 'sk-ant-fixture-worker-only')
    local_store = worker / 'auth.json'
    before = local_store.read_bytes() if local_store.exists() else None
    for provider in (PROVIDER, 'chatgpt'):
        with pytest.raises(auth.AuthError, match='Invalid shared provider credential policy'):
            resolve_runtime_with_fallback(config(fallback_providers=[{'provider': 'anthropic', 'model': 'other'}]),
                                          requested=provider)
    assert auth._global_auth_file_path() is None
    assert (local_store.read_bytes() if local_store.exists() else None) == before


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


def test_shareable_list_prefers_canonical_singleton_over_mirrors(stores, monkeypatch, capsys):
    from argparse import Namespace
    from hermes_cli.auth_commands import auth_list_command

    root, _ = stores
    canonical = row(source='device_code')
    mirror = dict(canonical, id='mirror', source='manual:device_code')
    stale = dict(mirror, id='stale', access_token=token(generation=0), refresh_token='old')
    independent = row('independent', 'other-account')
    seed(root, mirror, stale, canonical, independent,
         singleton={k: canonical[k] for k in ('access_token', 'refresh_token')})
    monkeypatch.setenv('HERMES_HOME', str(root))
    before = (root / 'auth.json').read_bytes()
    auth_list_command(Namespace(provider=PROVIDER, shareable=True))
    assert capsys.readouterr().out.splitlines() == [f'id={ROW} account_id={ACCOUNT} subject={SUBJECT}',
                                                    f'id=independent account_id=other-account subject={SUBJECT}']
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


def test_processes_share_exactly_one_root_refresh(stores, tmp_path):
    root, worker = stores
    subject = 'fixture-federation|fixture-member'
    started = row(source='device_code', access_token=token(subject=subject, expired=True))
    unrelated = row('other', 'other-account', priority=1)
    expected = token(subject=subject, generation=2)
    seed(root, started, unrelated, singleton={k: started[k] for k in ('access_token', 'refresh_token')})
    atomic_config_write(root / 'config.yaml', {'model': {'provider': PROVIDER}})
    settings = config()
    settings['auth']['shared_providers'][PROVIDER]['subject'] = subject
    atomic_config_replace(worker / 'config.yaml', settings)
    calls = []
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *_args):
            pass

        def do_POST(self):
            fields = parse_qs(self.rfile.read(int(self.headers['Content-Length'])).decode())
            with lock:
                calls.append(self.path == '/oauth/token' and fields.get('refresh_token') == ['fixture-refresh'])
                accepted = len(calls) == 1 and calls[-1]
            time.sleep(.7)  # Keep the stale generation visible during contention.
            payload = ({'access_token': expected, 'refresh_token': 'fixture-rotated'} if accepted else
                       {'error': 'invalid_grant', 'error_description': 'refresh_token_reused'})
            content = json.dumps(payload).encode()
            self.send_response(200 if accepted else 400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    child = '''
import os, sys, time
from pathlib import Path
import hermes_cli.auth_codex as codex
from hermes_cli.runtime_provider import resolve_runtime_provider
from agent.credential_pool import load_pool

import httpx
send = httpx.Client.send
def loopback_only(self, request, *args, **kwargs):
    assert str(request.url) == os.environ['TOKEN_URL']
    return send(self, request, *args, **kwargs)
httpx.Client.send = loopback_only
codex.CODEX_OAUTH_TOKEN_URL = os.environ['TOKEN_URL']
index, ready, go = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pool = load_pool('openai-codex') if index >= 8 else None
ready.touch()
deadline = time.monotonic() + 25
while not go.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('parent fixture start barrier')
    time.sleep(.01)
if pool is None:
    runtime = resolve_runtime_provider(requested='openai-codex')
    assert runtime['source'] == 'shared-root-oauth'
    selected = runtime['api_key']
else:
    entry = pool.try_refresh_matching(api_key_hint=os.environ['OLD'])
    selected = entry.access_token if entry else None
assert selected == os.environ['EXPECTED'], 'process did not adopt the one refreshed generation'
print('generation verified')
'''
    checkout = Path(auth.__file__).resolve().parents[1]
    barrier = tmp_path / 'go'
    readies = [tmp_path / f'ready-{i}' for i in range(10)]
    with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server, ExitStack() as stack:
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        stack.callback(server.shutdown)
        stack.callback(lambda: barrier.touch(exist_ok=True))
        url = f'http://127.0.0.1:{server.server_port}/oauth/token'
        processes = []
        for i, ready in enumerate(readies):
            profile = worker if i < 8 else root
            env = {'HOME': str(tmp_path), 'HERMES_HOME': str(profile), 'TMPDIR': str(tmp_path),
                   'PATH': os.defpath, 'PYTHONPATH': str(checkout), 'HERMES_DISABLE_LAZY_INSTALLS': '1',
                   'TOKEN_URL': url, 'OLD': started['access_token'], 'EXPECTED': expected}
            proc = subprocess.Popen([sys.executable, '-BP', '-c', child, str(i), str(ready), str(barrier)],
                                    cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def cleanup(p=proc):
                if p.poll() is None:
                    p.kill()
                p.communicate(timeout=10)

            stack.callback(cleanup)
            processes.append(proc)
        deadline = time.monotonic() + 18
        while not all(p.exists() for p in readies) and time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                break
            time.sleep(.02)
        assert all(p.exists() for p in readies), 'a fixture child did not reach the start barrier'
        barrier.touch()
        outcomes = [p.communicate(timeout=28) for p in processes]
        assert calls == [True], f'{len(calls)} refresh attempts for one grant'
        assert all(p.returncode == 0 and out.strip() == 'generation verified'
                   for p, (out, _err) in zip(processes, outcomes)), 'not all children adopted the refreshed grant'
    store = json.loads((root / 'auth.json').read_text())
    by_id = {entry['id']: entry for entry in store['credential_pool'][PROVIDER]}
    assert by_id[ROW]['access_token'] == store['providers'][PROVIDER]['tokens']['access_token'] == expected
    assert by_id[ROW]['refresh_token'] == 'fixture-rotated'
    assert by_id['other']['access_token'] == unrelated['access_token']
    assert by_id['other']['refresh_token'] == unrelated['refresh_token']
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


def test_federated_subject_lists_and_resolves_exact_singleton(stores, monkeypatch):
    from hermes_cli.auth_codex import list_shareable_codex_grants

    root, worker = stores
    subject = 'fixture-federation|fixture-member'
    access = token(subject=subject)
    selected = row(source='device_code')
    selected['access_token'] = access
    singleton = {'access_token': access, 'refresh_token': selected['refresh_token']}
    seed(root, selected, singleton=singleton)
    monkeypatch.setenv('HERMES_HOME', str(root))
    refs = list_shareable_codex_grants()
    assert [(ref.credential_id, ref.account_id, ref.subject) for ref in refs] == [(ROW, ACCOUNT, subject)]

    monkeypatch.setenv('HERMES_HOME', str(worker))
    settings = config()
    configured_ref = settings['auth']['shared_providers'][PROVIDER]
    configured_ref['subject'] = subject
    atomic_config_replace(worker / 'config.yaml', settings)
    assert resolve_runtime_provider(requested=PROVIDER)['source'] == 'shared-root-oauth'
    configured_ref['subject'] = 'different-federation|fixture-member'
    atomic_config_replace(worker / 'config.yaml', settings)
    with pytest.raises(auth.AuthError, match='Referenced root Codex OAuth grant'):
        resolve_runtime_provider(requested=PROVIDER)

    # A provider claim must match literally: trimming it must not manufacture a match.
    selected['access_token'] = singleton['access_token'] = token(subject=f' {subject}')
    seed(root, selected, singleton=singleton)
    monkeypatch.setenv('HERMES_HOME', str(root))
    assert list_shareable_codex_grants() == []
    monkeypatch.setenv('HERMES_HOME', str(worker))
    configured_ref['subject'] = subject
    atomic_config_replace(worker / 'config.yaml', settings)
    with pytest.raises(auth.AuthError, match='Referenced root Codex OAuth grant'):
        resolve_runtime_provider(requested=PROVIDER)
    assert not (worker / 'auth.json').exists()


@pytest.mark.parametrize(('field', 'value'), [
    ('account_id', 'issuer|member'), ('credential_id', 'issuer|member'),
    ('subject', 'issuer|other|member'), ('subject', 'issuer|../member'),
    ('subject', 'issuer|member:other'), ('subject', 'issuer|'),
    ('subject', '|member'), ('subject', 'issuer|member\n'), ('subject', 'issuer| leading-space'),
    ('subject', 'i|' + 'm' * 127),
])
def test_subject_namespace_does_not_relax_reference_validation(field, value):
    from hermes_cli.auth_shared_providers import parse_shared_provider_refs

    settings = config()
    settings['auth']['shared_providers'][PROVIDER][field] = value
    with pytest.raises(auth.AuthError, match='Invalid shared provider credential policy'):
        parse_shared_provider_refs(settings)
