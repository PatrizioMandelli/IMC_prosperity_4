import webbrowser
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any


class HTTPRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.server.shutdown_flag = True
        return super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        return super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


class CustomHTTPServer(HTTPServer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shutdown_flag = False


def open_visualizer(output_file: Path) -> None:
    http_handler = partial(
        HTTPRequestHandler, directory=str(output_file.parent))
    http_server = CustomHTTPServer(("localhost", 0), http_handler)

    print(
        f"\n[!] Custom visualizer logic will go here. Log saved to: {output_file.name}")
    # webbrowser.open(...)

    while not http_server.shutdown_flag:
        http_server.handle_request()
