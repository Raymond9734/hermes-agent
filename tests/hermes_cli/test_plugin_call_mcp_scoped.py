"""The scoped SDK call uses a real HTTP MCP session, not a registration filter."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_constants import get_hermes_home
from tests.waiting import wait_until

TOOLS = ["whoami", "get_issue", "get_dev_work", "get_ai_tasks", "get_dev_context"]

_SERVER = '''
import asyncio
import json
import socket
import sys
from pathlib import Path
from urllib.parse import parse_qs
import uvicorn
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPError
from mcp.types import ToolAnnotations

port, events = map(Path, sys.argv[1:])
server = MCPServer("fixture")

async def reply(ctx: Context):
    session = ctx.request_context.session
    caps = session.client_capabilities
    callbacks = [caps.sampling is not None, caps.elicitation is not None]
    refused = []
    if not any(callbacks):
        # A server normally checks capabilities before issuing these. Send both anyway.
        for method, params in [
            ("sampling/createMessage", {"messages": [{"role": "user", "content": {"type": "text", "text": "fixture"}}], "maxTokens": 1}),
            ("elicitation/create", {"message": "fixture", "requestedSchema": {"type": "object", "properties": {}}}),
        ]:
            try:
                await session._request_outbound.send_raw_request(method, params, {"timeout": 2})
            except MCPError as error:
                refused.append(error.error.message)
    return {"callbacks": callbacks, "refused": refused}

for name in ["whoami", "get_issue", "get_dev_work", "get_ai_tasks", "get_dev_context", "manage_dev_issues"]:
    server.add_tool(reply, name=name, annotations=ToolAnnotations(readOnlyHint=name != "manage_dev_issues"))
app = server.streamable_http_app()

async def record(scope, receive, send):
    if scope["type"] != "http":
        return await app(scope, receive, send)
    with events.open("a") as f:
        print(scope["method"], scope["path"], file=f)
    if scope["path"] == "/token":
        form = parse_qs((await receive())["body"].decode())
        if scope["method"] != "POST" or any(form.get(key) != [value] for key, value in {
                "grant_type": "refresh_token", "refresh_token": "fixture-refresh-only",
                "client_id": "fixture-client"}.items()):
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": json.dumps({
            "access_token": "fixture-synthetic-only", "token_type": "Bearer", "expires_in": 3600,
        }).encode()})
        return
    headers = dict(scope["headers"])
    if (events.with_suffix(".reject").exists() or not (
            headers.get(b"authorization") == b"Bearer fixture-synthetic-only"
            or headers.get(b"x-fixture") == b"fixture-header-only")):
        reject = events.with_suffix(".reject")
        status = int(reject.read_text()) if reject.exists() else 401
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"www-authenticate", b'Bearer error="insufficient_scope", scope="other"')]})
        await send({"type": "http.response.body", "body": b""})
        return
    await app(scope, receive, send)

async def main():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    uv = uvicorn.Server(uvicorn.Config(record, log_level="warning"))
    task = asyncio.create_task(uv.serve(sockets=[sock]))
    while not uv.started:
        await asyncio.sleep(.01)
    port.write_text(str(sock.getsockname()[1]))
    await task

asyncio.run(main())
'''


@pytest.fixture
def connector(tmp_path):
    script, port, events = (tmp_path / n for n in ("mcp_fixture.py", "port", "events"))
    script.write_text(_SERVER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(script), str(port), str(events)])
    try:
        wait_until(lambda: port.exists() and port.read_text(), timeout=10, what="fixture MCP bind")
        yield f"http://127.0.0.1:{port.read_text()}/mcp", events
    finally:
        proc.terminate()
        proc.wait(10)


def _configure(url, *, hosted=False):
    home = get_hermes_home()
    plugin = home / "plugins" / "owner-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(json.dumps({"name": "owner-fixture", "mcp_scopes": {"locus": TOOLS}}))
    (plugin / "__init__.py").write_text('''
import json, os

def register(ctx):
    def probe(args, **kw):
        data = ctx.call_mcp_scoped("locus", args["name"], timeout=5)
        return json.dumps({"pid": os.getpid(), "data": data})
    ctx.register_tool(name="scoped_fixture", toolset="fixture",
                      schema={"name": "scoped_fixture", "description": "Fixture only",
                              "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}},
                      handler=probe)
''')
    cfg = {
        "plugins": {"enabled": ["owner-fixture"], "isolation": "host" if hosted else "in_process", "entries": {
            "owner-fixture": {"mcp_scoped_allowlist": {"locus": TOOLS}},
            "ordinary-fixture": {"mcp_allowlist": ["locus"]},
        }},
        "mcp_servers": {"locus": {"url": url, "auth": "oauth", "connect_timeout": 5}},
    }
    (home / "config.yaml").write_text(json.dumps(cfg))
    return plugin, cfg


def _invoke(source):
    result = subprocess.run([sys.executable, "-B", "-c", source], cwd=Path(__file__).resolve().parents[2],
                            env={**os.environ, "HERMES_DISABLE_LAZY_INSTALLS": "1"},
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


_CLIENT = '''
import json
from hermes_constants import get_hermes_home
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_cli.plugins_manifest import parse_manifest_file

home = get_hermes_home()
plugin = home / "plugins" / "owner-fixture"
owner = PluginContext(parse_manifest_file(plugin / "plugin.yaml", plugin, "user", ""), PluginManager())
ordinary = PluginContext(PluginManifest(name="ordinary-fixture"), PluginManager())
'''


def test_scope_denials_do_not_connect(connector):
    url, events = connector
    _configure(url)
    _invoke(_CLIENT + '''
from pathlib import Path

def deny(ctx, server, name):
    try:
        ctx.call_mcp_scoped(server, name)
    except PermissionError:
        return
    raise AssertionError(f"accepted {server}/{name}")

for name in ("manage_dev_issues", "mcp__locus__whoami", "locus/whoami", "whoami ", "WHOAMI", "query"):
    deny(owner, "locus", name)
deny(owner, "mcp__locus", "whoami")
deny(ordinary, "locus", "whoami")
from hermes_cli.plugins_loader import _plugin_home_scope
with _plugin_home_scope(home / "other-profile"):
    deny(owner, "locus", "whoami")
try:
    owner.call_mcp("locus", "manage_dev_issues")
except PermissionError:
    pass
else:
    raise AssertionError("scoped grant also gave legacy access")
file = home / "config.yaml"
cfg = json.loads(file.read_text())
grant = cfg['plugins']['entries']['owner-fixture']['mcp_scoped_allowlist']
grant['locus'].append('manage_dev_issues')
file.write_text(json.dumps(cfg))
deny(owner, "locus", "whoami")
grant['locus'].pop()
cfg['plugins']['entries']['owner-fixture']['mcp_allowlist'] = ['locus']
file.write_text(json.dumps(cfg))
deny(owner, "locus", "whoami")
del cfg['plugins']['entries']['owner-fixture']['mcp_allowlist']
cfg['mcp_servers']['locus']['command'] = 'must-not-spawn'
file.write_text(json.dumps(cfg))
deny(owner, "locus", "whoami")
del cfg['mcp_servers']['locus']['command']
file.write_text('plugins: [ malformed')
deny(owner, "locus", "whoami")
cfg['plugins']['enabled'] = []
file.write_text(json.dumps(cfg))
deny(owner, "locus", "whoami")
''')
    assert not events.exists()


def _seed_oauth(url, *, expired=False, metadata=False, issuer_matches=True):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken
    from tools.mcp_oauth_provider import prepare_oauth_config

    _, storage = prepare_oauth_config("locus", url, None)
    issuer = url.removesuffix("/mcp")
    if metadata:
        storage.save_oauth_metadata(OAuthMetadata.model_validate({
            "issuer": issuer, "token_endpoint": f"{issuer}/token",
            "authorization_endpoint": f"{issuer}/authorize", "response_types_supported": ["code"],
        }))
    if expired:
        storage.bind_issuer(issuer if issuer_matches else f"{issuer}/previous")
        asyncio.run(storage.set_client_info(OAuthClientInformationFull.model_validate({
            "client_id": "fixture-client", "redirect_uris": [f"{issuer}/callback"], "token_endpoint_auth_method": "none",
        })))
    asyncio.run(storage.set_tokens(OAuthToken(
        access_token="fixture-expired-only" if expired else "fixture-synthetic-only", token_type="Bearer",
        expires_in=0 if expired else None, refresh_token="fixture-refresh-only" if expired else None)))
    return storage


def test_scoped_oauth_session_refuses_callbacks_and_does_not_replace_ordinary(connector):
    url, events = connector
    _configure(url, hosted=True)
    storage = _seed_oauth(url)
    check = _CLIENT + f'\nevents = __import__("pathlib").Path({str(events)!r})\n' + '''
import os
from tools.mcp_tool_discovery import discover_mcp_tools, _get_connected_server_for_call
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
from tools.mcp_oauth_manager import get_manager
from tools.registry import registry

manager = PluginManager()
try:
    discover_mcp_tools(allowed_mcp_names=['locus'])
    shared = _get_connected_server_for_call('locus')
    assert shared is not None and shared.session is not None
    before = shared.session
    manager.discover_and_load()
    assert manager._plugins['owner-fixture'].error is None
    mark = len(events.read_text().splitlines())
    expected = sorted(json.loads((home / 'config.yaml').read_text())['plugins']['entries']['owner-fixture']['mcp_scoped_allowlist']['locus'])
    for name in expected:
        reply = json.loads(registry.dispatch('scoped_fixture', {'name': name}, scope=manager.scope_key))
        assert reply['pid'] != os.getpid(), reply
        result = reply['data']
        assert result['ok'] is True, result
        assert result['scope'] == dict(plugin='owner-fixture', server='locus', tools=expected, sampling=False, elicitation=False)
        value = result.get('structuredContent') or json.loads(result['result'])
        assert value['callbacks'] == [False, False], value
        assert value['refused'] == ['Sampling not supported', 'Elicitation not supported'], value
    scoped_events = events.read_text().splitlines()[mark:]
    assert scoped_events and {line.split()[1] for line in scoped_events} == {'/mcp'}, scoped_events
    assert _get_connected_server_for_call('locus').session is before
    result = ordinary.call_mcp('locus', 'manage_dev_issues')
    value = result.get('structuredContent') or json.loads(result['result'])
    assert value['callbacks'] == [True, True], value
    assert set(get_manager()._entries) == {(str(home.resolve()), 'locus')}
finally:
    manager.unload()
    manager._plugin_host().shutdown()
    shutdown_mcp_servers()
'''
    for _ in range(2):  # fresh Hermes processes, not client objects sharing an in-memory grant
        assert "fixture-synthetic-only" not in _invoke(check)
    assert storage.has_cached_tokens()


@pytest.mark.parametrize("status", [401, 403])
def test_scoped_oauth_does_not_start_login(connector, status):
    url, events = connector
    _configure(url)
    check = _CLIENT + '''
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
try:
    reply = owner.call_mcp_scoped('locus', 'whoami', timeout=5)
    assert reply['ok'] is False and 'scope' not in reply, reply
finally:
    shutdown_mcp_servers()
'''
    _invoke(check)
    assert not events.exists()
    storage = _seed_oauth(url)
    events.with_suffix('.reject').write_text(str(status))
    _invoke(check)
    assert events.read_text().splitlines() == ['POST /mcp']
    assert storage.has_cached_tokens()


@pytest.mark.parametrize("grant", ["live", "bound", "missing_metadata", "wrong_issuer"])
def test_scoped_oauth_cold_process_only_uses_existing_authorization(connector, grant):
    url, events = connector
    _configure(url)
    expected_ok = grant in {"live", "bound"}
    storage = _seed_oauth(url, expired=grant != "live", metadata=grant in {"bound", "wrong_issuer"},
                          issuer_matches=grant != "wrong_issuer")
    saved = {path: path.read_bytes() for path in storage._state_paths() if path.exists()}
    _invoke(_CLIENT + f'\nexpected_ok = {expected_ok!r}\n' + '''
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
try:
    reply = owner.call_mcp_scoped('locus', 'whoami', timeout=5)
    assert reply['ok'] is expected_ok and ('scope' in reply) is expected_ok, reply
finally:
    shutdown_mcp_servers()
''')
    if expected_ok:
        lines = events.read_text().splitlines()
        if grant == "bound":
            assert lines.pop(0) == "POST /token"
            assert storage._tokens_path().read_bytes() != saved.pop(storage._tokens_path())
        assert lines[0] == "POST /mcp" and {line.split()[1] for line in lines} == {"/mcp"}
    else:
        assert not events.exists()
    assert all(path.read_bytes() == before for path, before in saved.items())


def test_scoped_resolves_existing_header_reference_and_revocation(connector):
    url, events = connector
    _, cfg = _configure(url)
    home = get_hermes_home()
    cfg['mcp_servers']['locus'].pop('auth')
    cfg['mcp_servers']['locus']['headers'] = {'x-fixture': '${SCOPED_MCP_FIXTURE_HEADER}'}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (home / '.env').write_text('SCOPED_MCP_FIXTURE_HEADER=fixture-header-only\n')
    _invoke(_CLIENT + f'\nevents = __import__("pathlib").Path({str(events)!r})\n' + '''
from tools.mcp_tool_discovery import _get_connected_server_for_call
from tools.mcp_tool_lifecycle import shutdown_mcp_servers
try:
    reply = owner.call_mcp_scoped('locus', 'whoami', timeout=5)
    assert reply['ok'] is True and reply['scope']['server'] == 'locus', reply
    assert _get_connected_server_for_call('locus') is None
    before = events.read_text()
    file = home / 'config.yaml'
    cfg = json.loads(file.read_text())
    cfg['plugins']['entries']['owner-fixture'].pop('mcp_scoped_allowlist')
    file.write_text(json.dumps(cfg))
    try:
        owner.call_mcp_scoped('locus', 'whoami', timeout=5)
    except PermissionError:
        pass
    else:
        raise AssertionError('stale grant accepted')
    assert events.read_text() == before
finally:
    shutdown_mcp_servers()
''')
