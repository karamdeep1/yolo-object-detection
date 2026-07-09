import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "runs/detect/train3/weights/best.pt"
DEFAULT_SOURCE = "images/canopyTentPeople.webp"
FAST_MAX_DET = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Run canopy tent/person detection.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to a .pt, .onnx, .engine, or exported model folder.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image, video, directory, or camera source.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--device", default="0", help='Device to use, such as "0" for GPU or "cpu".')
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported.")
    parser.add_argument("--fast", action="store_true", help="Use speed-focused defaults for airborne inference.")
    parser.add_argument("--save", action="store_true", help="Save annotated prediction images.")
    parser.add_argument("--stream", action="store_true", help="Stream results for video/camera sources.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Run inference on every Nth video frame.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image/frame.")
    parser.add_argument(
        "--classes",
        help='Optional comma-separated class IDs to detect, such as "1" for tents only.',
    )
    parser.add_argument("--benchmark", action="store_true", help="Run repeated inference and report average speed.")
    parser.add_argument("--runs", type=int, default=20, help="Benchmark inference runs.")
    parser.add_argument("--warmup", type=int, default=3, help="Benchmark warmup runs.")
    parser.add_argument("--api-url", help="HTTP endpoint that receives detection JSON.")
    parser.add_argument("--api-timeout", type=float, default=1.0, help="Seconds to wait for API POST responses.")
    parser.add_argument("--api-every", type=int, default=1, help="For streams, POST every Nth processed frame.")
    parser.add_argument("--json", action="store_true", help="Print detections as formatted JSON.")
    args = parser.parse_args()

    if args.fast:
        args.half = True
        args.save = False
        args.max_det = min(args.max_det, FAST_MAX_DET)

    if args.classes:
        args.classes = [int(class_id.strip()) for class_id in args.classes.split(",")]

    if args.api_every < 1:
        parser.error("--api-every must be 1 or greater")

    return args


def validate_path(path, label):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def is_live_or_remote_source(source):
    source = str(source)
    return source.isdigit() or source.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def resolve_model_path(model_path):
    model_path = Path(model_path)
    engine_path = model_path.with_suffix(".engine")

    if model_path == Path(DEFAULT_MODEL) and engine_path.exists():
        print(f"Using TensorRT engine for speed: {engine_path}")
        return engine_path

    return validate_path(model_path, "Model")


def resolve_source(source):
    if is_live_or_remote_source(source):
        return source

    return validate_path(source, "Source")


def load_model(model_path):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed in this Python environment. "
            "Install it with: pip install -U ultralytics"
        ) from exc

    return YOLO(str(model_path))


def post_json(url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"API POST failed: {exc}")
        return False


def predict(model, args, verbose=False):
    return model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        save=args.save,
        stream=args.stream,
        vid_stride=args.vid_stride,
        max_det=args.max_det,
        classes=args.classes,
        verbose=verbose,
    )


def collect_detections(results):
    detections = []

    for result in results:
        image_detections = {
            "source": str(getattr(result, "path", "")),
            "detections": [],
        }

        for box in result.boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            image_detections["detections"].append(
                {
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "confidence": round(confidence, 4),
                    "bbox_xyxy": [round(value, 2) for value in (x1, y1, x2, y2)],
                }
            )

        detections.append(image_detections)

    return detections


def summarize_result(result):
    detections = collect_detections([result])[0]
    counts = {}

    for detection in detections["detections"]:
        class_name = detection["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1

    count_text = ", ".join(f"{name}: {count}" for name, count in counts.items())
    return count_text or "no detections"


def run_once(model, args):
    results = predict(model, args)
    detections = collect_detections(results)
    payload = {
        "type": "image_batch",
        "timestamp": time.time(),
        "model": str(args.model),
        "source": str(args.source),
        "images": detections,
    }

    if args.json:
        print(json.dumps(detections, indent=2))
    else:
        for image in detections:
            print(f"{image['source']}: {len(image['detections'])} detections")
            for detection in image["detections"]:
                print(
                    f"  {detection['class_name']} "
                    f"{detection['confidence']:.2f} "
                    f"{detection['bbox_xyxy']}"
                )

    if args.api_url:
        post_json(args.api_url, payload, args.api_timeout)

    return detections


def run_stream(model, args):
    results = predict(model, args)
    frame_count = 0
    start = time.perf_counter()

    for result in results:
        frame_count += 1
        elapsed = time.perf_counter() - start
        fps = frame_count / elapsed if elapsed else 0
        detections = collect_detections([result])[0]
        print(f"Frame {frame_count}: {summarize_result(result)} | {fps:.2f} FPS")

        if args.api_url and frame_count % args.api_every == 0:
            payload = {
                "type": "frame",
                "timestamp": time.time(),
                "frame": frame_count,
                "fps": round(fps, 2),
                "model": str(args.model),
                "source": str(args.source),
                "image": detections,
            }
            post_json(args.api_url, payload, args.api_timeout)

    return frame_count


def benchmark(model, args):
    args.save = False
    args.stream = False

    for _ in range(args.warmup):
        predict(model, args)

    start = time.perf_counter()
    for _ in range(args.runs):
        results = predict(model, args)
    elapsed = time.perf_counter() - start

    avg_time = elapsed / args.runs
    fps = 1 / avg_time if avg_time else 0
    detections = collect_detections(results)
    detection_count = sum(len(image["detections"]) for image in detections)

    print(f"Runs: {args.runs}")
    print(f"Average time per run: {avg_time:.4f} s")
    print(f"Average FPS: {fps:.2f}")
    print(f"Detections in final run: {detection_count}")

    return detections


def main():
    args = parse_args()
    args.model = resolve_model_path(args.model)
    args.source = resolve_source(args.source)

    model = load_model(args.model)

    if args.benchmark:
        return benchmark(model, args)

    if args.stream:
        return run_stream(model, args)

    return run_once(model, args)


if __name__ == "__main__":
    main()
