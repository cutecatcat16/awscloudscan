#!/usr/bin/env python3
"""Direct entry point so the wrapper can run without installation.

``install.sh`` puts a launcher on your PATH as a *symlink* to this file, so when
``awscloudscan`` is run the ``__file__`` we see is the path taken through the
symlink — always ``~/.local/bin/...``, never the checkout. Deriving ``sys.path``
from that would make ``import cloudscan`` look for the package beside the
symlink, which is why running the command from another directory used to fail
with ``ModuleNotFoundError: No module named 'cloudscan'``. ``realpath`` collapses
the symlink to the real script, so the package is found wherever the checkout
actually lives.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from cloudscan.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
