"""Keep full setup output private while streaming only known progress counters."""
import re
import sys
import threading
from pathlib import Path


PROGRESS = re.compile(
    r'(?:Installing recommendation outbox triggers|Indexing recent posts|Importing recent interactions|'
    r'catalog bootstrap batch=\d+|history source=[A-Za-z]+ enqueued|'
    r'replay batch=\d+ last_id=\d+ high_water=\d+|'
    r'Recommendation setup complete: pending=\d+ dead=\d+)'
)


def stream(source, log_path, emit=print, interval=30):
    done = threading.Event()

    def heartbeat():
        seconds = 0
        while not done.wait(interval):
            seconds += interval
            emit(f'Setup process still running ({seconds}s elapsed); waiting for the next progress batch.', flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with Path(log_path).open('a') as log:
            for line in source:
                log.write(line)
                log.flush()
                match = PROGRESS.search(line)
                if match:
                    emit(match.group(0), flush=True)
    finally:
        done.set()
        thread.join()


if __name__ == '__main__':
    stream(sys.stdin, sys.argv[1])
