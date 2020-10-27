"""Redirecting writing to tqdm (the progressbar).

See here: https://github.com/tqdm/tqdm#redirecting-writing
"""

import contextlib
import sys
from tqdm import tqdm


class DummyTqdmFile:
    """Dummy file-like that will write to tqdm."""

    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

    def close(self):
        return getattr(self.file, "close", lambda: None)()

    def isatty(self):
        return getattr(self.file, "isatty", lambda: False)()


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
