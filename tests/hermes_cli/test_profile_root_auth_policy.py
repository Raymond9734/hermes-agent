"""Named-profile auth policy against two real, disposable Hermes stores."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import hermes_constants
from hermes_cli import auth
from hermes_cli.config import FailedConfigRead, atomic_config_write, load_config_readonly


@pytest.fixture
def auth_stores(tmp_path, monkeypatch):
    fake_home = tmp_path / 'home'
    fake_home.mkdir()
    root = tmp_path / 'hermes-root'
    profile = root / 'profiles' / 'worker'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HOME', str(fake_home))
    monkeypatch.setenv('HERMES_HOME', str(profile))
    monkeypatch.setattr(Path, 'home', lambda: fake_home)
    monkeypatch.setattr(hermes_constants, '_default_hermes_root_memo', None)
    row = {'id': 'root-fixture', 'source': 'manual', 'access_token': 'fake-root-credential'}
    store = {'version': 1, 'credential_pool': {'anthropic': [row]},
             'providers': {'openai-codex': {'tokens': {'access_token': 'fake-root-oauth'}}}}
    (root / 'auth.json').write_text(json.dumps(store), encoding='utf-8')
    return root, profile, row


def test_isolated_profile_does_not_resolve_root_credentials(auth_stores):
    _, profile, _ = auth_stores
    atomic_config_write(profile / 'config.yaml', {'auth': {'inherit_root_credentials': False}})

    assert auth.read_credential_pool('anthropic') == []
    assert auth.read_credential_pool() == {}
    assert auth.get_provider_auth_state('openai-codex') is None


def test_default_profile_can_still_borrow_root_credentials(auth_stores):
    _, _, row = auth_stores
    assert auth.read_credential_pool('anthropic') == [row]
    assert auth.get_provider_auth_state('openai-codex') is not None


def test_isolated_profile_owns_its_credentials(auth_stores):
    _, profile, _ = auth_stores
    atomic_config_write(profile / 'config.yaml', {'auth': {'inherit_root_credentials': False}})
    row = {'id': 'worker-fixture', 'source': 'manual', 'access_token': 'fake-worker-credential'}
    auth.write_credential_pool('anthropic', [row])
    auth._save_active_provider_state('openai-codex', {'tokens': {'access_token': 'fake-worker-oauth'}})

    assert auth.read_credential_pool('anthropic') == [row]
    assert auth.get_provider_auth_state('openai-codex') == {'tokens': {'access_token': 'fake-worker-oauth'}}


@pytest.mark.parametrize('broken_yaml', ['auth: [missing-bracket\n', '- invalid top level\n', '[]\n', 'false\n'])
def test_unreadable_named_profile_never_borrows_root(auth_stores, broken_yaml):
    root, profile, _ = auth_stores
    (root / '.anthropic_oauth.json').write_text('{}', encoding='utf-8')
    (profile / 'config.yaml').write_text(broken_yaml, encoding='utf-8')
    assert not (profile / 'backups' / 'config').exists()

    assert auth.read_credential_pool('anthropic') == []
    assert auth.get_provider_auth_state('openai-codex') is None
    from agent.anthropic_credentials import _root_hermes_oauth_file
    assert _root_hermes_oauth_file() is None


@pytest.mark.parametrize('auth_section', [False, None, [], 'inherit_root_credentials: false'])
def test_malformed_auth_section_never_borrows_root(auth_stores, auth_section):
    _, profile, _ = auth_stores
    atomic_config_write(profile / 'config.yaml', {'auth': auth_section})

    assert auth.read_credential_pool('anthropic') == []


def test_unreadable_root_still_owns_its_credentials(auth_stores, monkeypatch):
    root, _, row = auth_stores
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(hermes_constants, '_default_hermes_root_memo', None)
    (root / 'config.yaml').write_text('auth: [missing-bracket\n', encoding='utf-8')

    assert isinstance(load_config_readonly(), FailedConfigRead)
    assert auth.read_credential_pool('anthropic') == [row]


@pytest.mark.parametrize('last_good_allows_root', [False, True])
def test_unreadable_named_profile_denies_root_even_with_last_good_backup(auth_stores, last_good_allows_root):
    root, profile, _ = auth_stores
    path = profile / 'config.yaml'
    atomic_config_write(path, {'auth': {'inherit_root_credentials': last_good_allows_root}})
    assert load_config_readonly()['auth']['inherit_root_credentials'] is last_good_allows_root
    assert any((profile / 'backups' / 'config').glob('*'))
    path.write_text('auth: [missing-bracket\n', encoding='utf-8')

    child = subprocess.run(
        [sys.executable, '-B', '-c',
         'import json; from hermes_cli import auth; '
         'from hermes_cli.config import FailedConfigRead, load_config_readonly; '
         'cfg = load_config_readonly(); '
         'print(json.dumps([isinstance(cfg, FailedConfigRead), '
         'cfg["auth"]["inherit_root_credentials"], auth.read_credential_pool("anthropic")]))'],
        env={'HOME': str(root.parent / 'home'), 'HERMES_HOME': str(profile),
             'PYTHONPATH': str(Path(__file__).resolve().parents[2])},
        cwd=root.parent, capture_output=True, text=True, check=True, timeout=20,
    )
    assert json.loads(child.stdout.splitlines()[-1]) == [True, last_good_allows_root, []]
