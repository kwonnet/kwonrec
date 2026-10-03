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


def test_setup_failure_reports_outside_redirection_and_restores_worker(tmp_path):
    """Exercise Bash's real errexit/function/redirection behavior without a VM."""
    import subprocess
    script = Path('deploy/compute/deploy.sh').read_text()
    wrapper = script[script.index('run_setup() {'):script.index('# Install/upgrade triggers each time;')]
    harness = '''set -Eeuo pipefail
WORK="$1"
dc() { echo 'captured setup exception'; return 23; }
restore_previous() { echo 'recovery invoked' >&2; cat "$WORK/setup.log" >&2; }
''' + wrapper + '\nrun_setup "Bootstrap probe"\necho UNEXPECTED_SUCCESS\n'
    result = subprocess.run(['bash', '-c', harness, 'test', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 23
    assert 'Bootstrap probe' in result.stderr
    assert 'captured setup exception' in result.stderr
    assert 'recovery invoked' in result.stderr
    assert 'UNEXPECTED_SUCCESS' not in result.stdout


def test_successful_setup_continues_without_recovery(tmp_path):
    import subprocess
    script = Path('deploy/compute/deploy.sh').read_text()
    wrapper = script[script.index('run_setup() {'):script.index('# Install/upgrade triggers each time;')]
    harness = '''set -Eeuo pipefail
WORK="$1"
dc() { echo 'setup complete'; }
restore_previous() { echo UNEXPECTED_RECOVERY; }
''' + wrapper + '\nrun_setup "Bootstrap probe"\necho SUCCESS\n'
    result = subprocess.run(['bash', '-c', harness, 'test', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'SUCCESS' in result.stdout
    assert 'UNEXPECTED_RECOVERY' not in result.stdout
