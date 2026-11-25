"""Only the loaded owner of live execution middleware can satisfy worker policy."""

import pytest

from hermes_cli import middleware, plugins


_CALLBACKS = """
def guard(**kwargs):
    payload = kwargs.get('request', kwargs.get('args'))
    if payload.get('fail'):
        raise RuntimeError('fixture guard refused')
    return kwargs['next_call']({**payload, 'checked': True})
"""


@pytest.fixture
def make_manager(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(middleware, '_worker_guards', {})

    def setup(**sources):
        for name, registration in sources.items():
            folder = home / 'plugins' / name
            folder.mkdir(parents=True)
            (folder / 'plugin.yaml').write_text(f'name: {name}\nversion: 0.1.0\n')
            (folder / '__init__.py').write_text(_CALLBACKS + '\ndef register(ctx):\n' + registration)
        (home / 'config.yaml').write_text(f'plugins:\n  enabled: {list(sources)!r}\n')
        manager = plugins.PluginManager()
        monkeypatch.setattr(plugins, 'get_plugin_manager', lambda: manager)
        manager.discover_and_load()
        return manager

    return setup


_REGISTER_LLM = "    ctx.register_middleware('llm_execution', guard)\n"
_REGISTER_TOOL = "    ctx.register_middleware('tool_execution', guard)\n"
_REGISTER_BOTH = _REGISTER_LLM + _REGISTER_TOOL


@pytest.mark.parametrize('registration', [
    None,
    _REGISTER_LLM,
    _REGISTER_LLM + "    ctx.register_middleware('tool_execution', None)\n",
    _REGISTER_BOTH + "    raise RuntimeError('registration failed')\n",
])
def test_other_plugin_cannot_supply_a_missing_or_failed_owners_guards(make_manager, registration):
    sources = {'other': _REGISTER_BOTH}
    if registration is not None:
        sources['required'] = registration
    make_manager(**sources)
    with pytest.raises(ValueError, match='required_worker_plugin_unavailable'):
        middleware.bind_required_worker_plugin('required')
    assert not middleware._worker_guards


@pytest.mark.parametrize('run', [middleware.run_llm_execution_middleware,
                                lambda args, callback: middleware.run_tool_execution_middleware('fixture', args, callback)])
def test_owned_guard_is_fail_closed_before_next_call_and_cannot_be_unloaded(make_manager, run):
    manager = make_manager(required=_REGISTER_BOTH)
    calls = []
    # Unbound execution middleware keeps its ordinary skip-on-error semantics.
    run({'fail': True}, calls.append)
    assert calls == [{'fail': True}]
    calls.clear()

    middleware.bind_required_worker_plugin('required')
    run({}, calls.append)
    assert calls == [{'checked': True}]
    with pytest.raises(RuntimeError, match='fixture guard refused'):
        run({'fail': True}, calls.append)
    assert manager.unload('required')
    with pytest.raises(ValueError, match='required_worker_plugin_unavailable'):
        run({}, calls.append)
    assert calls == [{'checked': True}]
