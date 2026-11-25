"""One-shot native MCP sessions for an explicitly granted plugin scope."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.safe_worker_policy import safe_worker_enabled
from hermes_cli.config import get_config_path
from hermes_cli.config_backend import ConfigBackendUnavailable, read_config_doc
from hermes_cli.managed_scope import apply_managed_overlay
from hermes_cli.plugins_manifest import parse_mcp_scopes
from hermes_cli.plugins_state import _plugin_settings_entry
from hermes_constants import hermes_home_key
from tools import mcp_tool as _core
from tools.mcp_oauth_manager import get_manager
from tools.mcp_tool_config import _interpolate_env_vars, _require_rendered_remote, _suspicious_mcp_issues
from tools.mcp_tool_costs import config_cost_policy
from tools.mcp_tool_discovery import _owner_secret_scope
from tools.mcp_tool_errors import _validate_remote_mcp_url
from tools.mcp_tool_handlers import _capped_structured_content, _render_call_tool_result
from tools.mcp_tool_loop import _ensure_mcp_loop, _run_on_mcp_loop
from tools.mcp_tool_registration import _annotation_read_only_hint, _normalize_server_trust
from utils import env_var_enabled


@dataclass(frozen=True)
class _Scope:
    home: str
    plugin: str
    server: str
    tools: frozenset[str]
    sampling: bool = False
    elicitation: bool = False

    def receipt(self) -> dict:
        return {"plugin": self.plugin, "server": self.server, "tools": sorted(self.tools),
                "sampling": self.sampling, "elicitation": self.elicitation}

    def authorize(self, abandoned: Callable[[], bool]) -> dict:
        """Read the current operator grant without config recovery or connector discovery."""
        if abandoned() or safe_worker_enabled() or env_var_enabled("HERMES_SAFE_MODE"):
            raise PermissionError("Scoped MCP is disabled in this context")
        if hermes_home_key() != self.home:
            raise PermissionError("Scoped MCP profile changed")
        try:
            raw = read_config_doc(get_config_path())
            cfg = apply_managed_overlay(raw, strict=True) if isinstance(raw, dict) else {}
        except (Exception, ConfigBackendUnavailable):
            raise PermissionError("Scoped MCP configuration unavailable") from None
        plugins = cfg.get("plugins")
        if not isinstance(plugins, dict):
            raise PermissionError("Scoped MCP plugin is not enabled")
        enabled, disabled = plugins.get("enabled"), plugins.get("disabled", [])
        if (not isinstance(enabled, list) or self.plugin not in enabled
                or not isinstance(disabled, list) or self.plugin in disabled):
            raise PermissionError("Scoped MCP plugin is not enabled")
        entry = _plugin_settings_entry(cfg, self.plugin) or {}
        if parse_mcp_scopes(entry.get("mcp_scoped_allowlist")).get(self.server) != self.tools:
            raise PermissionError("Scoped MCP grant does not match the plugin declaration")
        legacy = entry.get("mcp_allowlist", [])
        if not isinstance(legacy, list) or self.server in legacy:
            raise PermissionError("Scoped MCP cannot coexist with a whole-server grant for this plugin")
        servers = cfg.get("mcp_servers")
        connector = servers.get(self.server) if isinstance(servers, dict) else None
        if not isinstance(connector, dict) or connector.get("enabled", True) is not True:
            raise PermissionError("Scoped MCP connector is unavailable")
        if (not isinstance(connector.get("url"), str) or "command" in connector
                or connector.get("transport") not in (None, "streamable-http", "streamable_http")):
            raise PermissionError("Scoped MCP requires an existing Streamable HTTP connector")
        return connector


class _ScopedSession(_core.MCPServerTask):
    """Own just one native transport and SDK session; never publish it to the tool registry."""

    def __init__(self, scope: _Scope, config: dict):
        super().__init__(scope.server)  # Native auth key remains the original connector name.
        self.scope = scope             # Session identity, not the auth key or ordinary session key.
        self._config = config
        self._auth_type = (config.get("auth") or "").lower().strip()

    def _build_oauth_auth(self, url: str, config: dict):
        if self._auth_type == "oauth":
            return get_manager().get_existing_grant_auth(self.name, url, config.get("oauth"))
        return None

    async def call(self, tool: str, arguments: dict, recheck: Callable[[], None]) -> str:
        common, header_names = self._http_transport_options(self._config)
        url, _, timeout, _, _, auth, _ = common
        _validate_remote_mcp_url(self.name, url)
        if self._auth_type == "oauth" and auth is None:
            raise PermissionError("Scoped MCP connector authorization is unavailable")
        transport = self._streamable_http_transport(*common, header_names, follow_redirects=False)
        # Passing None selects the SDK's rejecting handlers without advertising either capability.
        # Nothing routes unsolicited requests to Hermes' model or approval handlers.
        async with transport as streams, _core.ClientSession(
            streams[0], streams[1], sampling_callback=None, elicitation_callback=None,
        ) as session:
            self.initialize_result = await self._negotiate_session(session, float(timeout))
            if not self._advertises_tools():
                return _failure("Scoped MCP connector does not advertise tools")
            tools = await _core._paginate_full_list(session.list_tools, "tools", self.name)
            granted = [item for item in tools if item.name in self.scope.tools]
            if len(granted) != len(self.scope.tools) or {item.name for item in granted} != self.scope.tools:
                return _failure("Scoped MCP connector does not advertise each granted tool exactly once")
            if (_normalize_server_trust(self._config.get("trust")) == _core._TRUST_UNTRUSTED
                    and not all(_annotation_read_only_hint(item) for item in granted)):
                return _failure("Scoped MCP cannot prompt for untrusted tools lacking a read-only annotation")
            await asyncio.to_thread(recheck)
            result = await session.call_tool(tool, arguments)
            if any(block.type != "text" for block in result.content):
                return _failure("Scoped MCP accepts only text and structured results")
            if isinstance(_capped_structured_content(result), str):
                return _failure("Scoped MCP structured result was too large")
            return _render_call_tool_result(result, self.name)


def _failure(message: str) -> str:
    return json.dumps({"error": message})


def call_scoped(home: str, plugin: str, server: str, tool: str, arguments: dict, declared: frozenset[str],
                timeout: float, *, abandoned: Callable[[], bool]) -> tuple[str, dict]:
    scope = _Scope(home, plugin, server, declared)
    source = scope.authorize(abandoned)
    if tool not in scope.tools:
        raise PermissionError("MCP tool is not in the active scope")

    def recheck() -> None:
        if scope.authorize(abandoned) != source:
            raise PermissionError("Scoped MCP connector changed before dispatch")

    async def invoke(config: dict) -> str:
        await asyncio.to_thread(recheck)
        return await _ScopedSession(scope, config).call(tool, arguments, recheck)

    try:
        timeout = max(1.0, min(float(timeout), 600.0))
    except (TypeError, ValueError):
        timeout = 30.0
    try:
        with _owner_secret_scope(Path(scope.home)):
            rendered = _interpolate_env_vars(source)
            if not isinstance(rendered, dict):
                return _failure("Scoped MCP connector configuration is unsupported"), {}
            config = _require_rendered_remote(server, rendered)
            if _suspicious_mcp_issues(server, config) or config_cost_policy(server, config) is not None:
                return _failure("Scoped MCP connector configuration is unsupported"), {}
            _validate_remote_mcp_url(server, config["url"])
            _ensure_mcp_loop()
            raw = _run_on_mcp_loop(lambda: invoke(config), timeout)
    except PermissionError:
        raise
    except (Exception, ConfigBackendUnavailable):
        # Connection exceptions can contain URLs, auth challenges or upstream response bodies.
        return _failure("Scoped MCP connector call failed"), {}
    return raw, scope.receipt()
