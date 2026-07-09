import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DetectionHandler(BaseHTTPRequestHandler):
    output_path = Path("latest_detections.json")

    def do_POST(self):
        if self.path != "/detections":
            self.send_error(404, "Use POST /detections")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Request body must be valid JSON")
            return

        DetectionHandler.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        detection_count = count_detections(payload)
        print(f"Received {payload.get('type', 'payload')} with {detection_count} detections")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, format, *args):
        return


def count_detections(payload):
    if "image" in payload:
        return len(payload["image"].get("detections", []))

    if "images" in payload:
        return sum(len(image.get("detections", [])) for image in payload["images"])

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Receive YOLO detection JSON on a Raspberry Pi.")
    parser.add_argument("--host", default="0.0.0.0", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--output", default="latest_detections.json", help="File to write the latest payload.")
    return parser.parse_args()


def main():
    args = parse_args()
    DetectionHandler.output_path = Path(args.output)

    server = ThreadingHTTPServer((args.host, args.port), DetectionHandler)
    print(f"Listening on http://{args.host}:{args.port}/detections")
    print(f"Writing latest payload to {DetectionHandler.output_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
