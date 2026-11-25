"""Named-profile auth policy against two real, disposable Hermes stores."""
import json
from pathlib import Path

import pytest

import hermes_constants
from hermes_cli import auth
from hermes_cli.config import atomic_config_write


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
