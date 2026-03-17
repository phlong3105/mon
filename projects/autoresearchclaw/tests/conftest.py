# conftest.py — shared pytest fixtures for researchclaw tests

import socket as _socket

# Fail-fast on any real network call during tests (2 s timeout).
# Properly mocked / patched calls are unaffected.
_socket.setdefaulttimeout(2)
