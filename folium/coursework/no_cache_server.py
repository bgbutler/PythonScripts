"""
no_cache_server.py
------------------
Drop-in replacement for `python -m http.server`.
Adds no-cache headers to every response so rebuilt map files
are always served fresh without needing a hard browser refresh.

Usage (from your notebook):
    subprocess.Popen(["python", "no_cache_server.py", "8000"], cwd=folder, ...)
"""

import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        pass  # suppress per-request logging


with socketserver.TCPServer(("", PORT), NoCacheHandler) as httpd:
    httpd.serve_forever()
