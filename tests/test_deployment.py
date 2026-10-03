import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path('deploy/compute') / f'{name}.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


BASE = 'KWONREC_API_KEY=' + 'x' * 32 + '\nKWONREC_DATABASE_URL=postgresql://u:p@db/kwonnet\nKWONREC_REDIS_URL=redis://redis:6379/0\n'


def test_runtime_target_is_stable_and_private():
    validate = module('validate-env').validate
    assert validate(BASE)[0] == '127.0.0.1'
    assert validate(BASE)[1] == validate(BASE + 'KWONREC_POLL_SECONDS=1\n')[1]
    assert validate(BASE)[1] != validate(BASE + 'KWONREC_NAMESPACE=new\n')[1]


@pytest.mark.parametrize('extra', ['KWONREC_BIND_IP=0.0.0.0', 'KWONREC_BIND_IP=8.8.8.8',
                                  'KWONREC_ALLOW_UNAUTHENTICATED=true', 'export WRONG=value',
                                  'KWONREC_API_KEY=duplicate'])
def test_rejects_unsafe_config(extra):
    with pytest.raises(ValueError):
        module('validate-env').validate(BASE + extra)


def test_rejects_container_local_database():
    with pytest.raises(ValueError, match='KWONREC_DATABASE_URL'):
        module('validate-env').validate(BASE.replace('@db/', '@localhost/'))


def test_diagnostics_redact_credentials():
    assert 'secret' not in module('diagnostics').redact('failed redis://u:secret@remote/0 secret', 'PASSWORD=secret')


@pytest.mark.parametrize('skip', [False, True])
def test_setup_backfills_only_when_requested(monkeypatch, skip):
    from src.runtime import setup
    monkeypatch.setattr('sys.argv', ['setup'] + (['--skip-bootstrap'] if skip else []))
    monkeypatch.setattr(setup, 'settings', lambda: SimpleNamespace(database_url='postgresql://u:p@db/test'))
    engine = MagicMock()
    monkeypatch.setattr(setup, 'Engine', lambda _: engine)
    connection = MagicMock()
    connection.execute.return_value.fetchone.return_value = {'pending': 0, 'dead': 0}
    context = MagicMock()
    context.__enter__.return_value = connection
    monkeypatch.setattr(setup.psycopg, 'connect', lambda *a, **kw: context)
    bootstrap = MagicMock()
    history = MagicMock()
    monkeypatch.setattr(setup, 'bootstrap', bootstrap)
    monkeypatch.setattr(setup, 'bootstrap_history', history)
    monkeypatch.setattr(setup, 'process_batch', lambda *a: 0)
    setup.main()
    assert bootstrap.call_count == (0 if skip else 1)
    assert history.call_count == (0 if skip else 1)
    assert 'CREATE SCHEMA' in connection.execute.call_args_list[0].args[0]
    engine.redis.close.assert_called_once()
