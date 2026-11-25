"""Validate named-profile references to individual root OAuth grants, not stores."""
from dataclasses import dataclass
import re
from typing import Any, TypeGuard

from hermes_cli.auth_constants import AuthError

# Eligibility requires a provider-specific implementation that can select and
# rotate one exact root-owned grant. Registry membership alone is not enough.
_SUPPORTED = frozenset({'openai-codex'})
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}')


class SharedProviderAuthError(AuthError):
    """A failed explicit credential reference must not enter the fallback ladder."""


@dataclass(frozen=True, slots=True)
class SharedProviderRef:
    credential_id: str
    account_id: str
    subject: str


def valid_reference_id(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def invalid_shared_policy() -> SharedProviderAuthError:
    return SharedProviderAuthError('Invalid shared provider credential policy.', code='invalid_shared_provider_policy')


def parse_shared_provider_refs(config: dict[str, Any]) -> dict[str, SharedProviderRef]:
    auth = config.get('auth', {})
    if not isinstance(auth, dict):
        raise invalid_shared_policy()
    refs = auth.get('shared_providers', {})
    if not isinstance(refs, dict):
        raise invalid_shared_policy()
    if refs and (auth.get('inherit_root_credentials') is not False
                 or auth.get('adopt_external_logins') is not False):
        raise invalid_shared_policy()
    parsed = {}
    for provider, ref in refs.items():
        if (provider not in _SUPPORTED or not isinstance(ref, dict)
                or ref.keys() != {'source', 'credential_id', 'account_id', 'subject'} or ref['source'] != 'root'
                or not all(valid_reference_id(ref[key]) for key in ('credential_id', 'account_id', 'subject'))):
            raise invalid_shared_policy()
        parsed[provider] = SharedProviderRef(ref['credential_id'], ref['account_id'], ref['subject'])
    return parsed


def active_shared_provider_refs(config: dict[str, Any] | None = None) -> dict[str, SharedProviderRef]:
    from agent.safe_worker_policy import worker_config_snapshot
    from hermes_cli.config_backend import ConfigBackendUnavailable
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_default_hermes_root, get_hermes_home

    home = get_hermes_home().resolve()
    root = get_default_hermes_root().resolve()
    if home == root:
        return {}  # Neither broken profile YAML nor a reference affects the primary's own login.
    if home.parent != root / 'profiles':
        return {}  # A reference cannot name another home or be used from an arbitrary directory.
    snapshot = worker_config_snapshot()
    try:
        effective = snapshot if snapshot is not None else config
        if effective is None:
            effective = load_user_config_effective(fail_closed=True)
    except (Exception, ConfigBackendUnavailable):  # health: allow BLE001 -- config I/O or parsing failure denies auth
        raise invalid_shared_policy() from None
    return parse_shared_provider_refs(effective)
