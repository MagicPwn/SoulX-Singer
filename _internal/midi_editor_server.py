import http.server
import os
import sys
import socket
import webbrowser
import time

PORT = 17888
HOST = "127.0.0.1"

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "midi-editor", "dist"))

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, f, *a):
        pass

server = http.server.HTTPServer((HOST, PORT), Handler)
print(f"SoulX-Singer MIDI Editor: http://{HOST}:{PORT}")

# Open browser after server is listening
webbrowser.open(f"http://{HOST}:{PORT}")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped.")
    server.shutdown()
