"""Bounded startup diagnostics, excluding runtime credentials."""
import pathlib
import re
import subprocess
import sys


def redact(text, environment):
    values = [line.partition('=')[2] for line in environment.splitlines()
              if '=' in line and not line.lstrip().startswith('#')]
    for value in sorted((v for v in values if len(v) >= 4), key=len, reverse=True):
        text = text.replace(value, '[REDACTED]')
    return re.sub(r'(?:postgres(?:ql)?|rediss?)://\S+', '[REDACTED_URL]', text)


if __name__ == '__main__':
    environment = pathlib.Path(sys.argv[1]).read_text()
    log = pathlib.Path(sys.argv[2])
    output = log.read_text()[-12000:] if log.exists() else ''
    for name in ('kwonrec-production-api-1', 'kwonrec-production-worker-1'):
        result = subprocess.run(['docker', 'logs', '--tail', '30', name], capture_output=True, text=True, check=False)
        output += '\n' + name + '\n' + result.stdout + result.stderr
    print(redact(output, environment))
