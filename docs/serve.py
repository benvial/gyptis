#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Benjamin Vial
# License: GPLv3


import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
DIRECTORY = sys.argv[2] if len(sys.argv) > 2 else "."

os.chdir(DIRECTORY)
socketserver.TCPServer.allow_reuse_address = True

with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
    print(f"Server running on http://localhost:{PORT}")
    httpd.serve_forever()
