"""Local HTTP acceptance test with a fixed workload, not a production SLA."""
import argparse
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
from statistics import median
import threading
import time
from urllib.request import Request, urlopen
import torch
from .inference import load_artifact
from .serve import handler_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer = load_artifact(args.artifact)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_type(model, tokenizer))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    times = []
    failures = 0
    example_text = None
    try:
        with urlopen(url + "/health") as response:
            health = json.load(response)
        for index in range(21):
            request = Request(url + "/generate", data=json.dumps(
                dict(prompt="The old man opened the door and", max_new_tokens=32)).encode(),
                headers={"Content-Type": "application/json"})
            began = time.perf_counter()
            try:
                with urlopen(request, timeout=15) as response:
                    text = json.load(response)["text"]
                    assert isinstance(text, str)
                    if example_text is None:
                        example_text = text
                if index:
                    times.append(time.perf_counter() - began)
            except (OSError, ValueError, AssertionError):
                if index:
                    failures += 1
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    ordered = sorted(times)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None
    report = dict(
        scope="local HTTP smoke acceptance; 20 sequential requests after 1 warmup",
        example_text=example_text,
        device="cpu", threads=2, output_tokens=32, health=health,
        prompt_tokens=len(tokenizer.encode("The old man opened the door and").ids),
        successes=len(times), failures=failures, availability_claim=False,
        e2e_median_seconds=median(times) if times else None, e2e_p95_seconds=p95,
        project_target_p95_seconds=2.0,
        passed=failures == 0 and p95 is not None and p95 <= 2.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
