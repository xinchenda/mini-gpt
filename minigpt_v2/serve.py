"""Small CPU inference API. python -m minigpt_v2.serve ARTIFACT"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock
import torch
from .inference import load_artifact


def handler_type(model, tokenizer):
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)

        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.send_json(404, {"error": "unknown route"})
                return
            self.send_json(200, {"status": "ready", "context": model.config.context})

        def do_POST(self):
            if self.path != "/generate":
                self.send_json(404, {"error": "unknown route"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Request body must be 1..16384 bytes")
                data = json.loads(self.rfile.read(size))
                prompt = data.get("prompt")
                count = data.get("max_new_tokens", 32)
                if not isinstance(prompt, str) or not prompt or len(prompt) > 4096:
                    raise ValueError("prompt must contain 1..4096 characters")
                if type(count) is not int or not 1 <= count <= 128:
                    raise ValueError("max_new_tokens must be an integer in 1..128")
                ids = tokenizer.encode(prompt).ids
                if not ids:
                    raise ValueError("Prompt encodes to no tokens")
                if not lock.acquire(blocking=False):
                    self.send_json(429, {"error": "generation busy"})
                    return
                try:
                    output = model.generate(torch.tensor([ids]), count, temperature=0)
                finally:
                    lock.release()
                self.send_json(200, {"text": tokenizer.decode(output[0].tolist())})
            except (ValueError, TypeError, AttributeError) as error:
                self.send_json(400, {"error": str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer = load_artifact(args.artifact)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_type(model, tokenizer))
    server.timeout = 30
    print(f"Listening on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
