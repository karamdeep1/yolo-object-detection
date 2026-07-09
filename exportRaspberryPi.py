import argparse
from pathlib import Path


DEFAULT_MODEL = "runs/detect/train3/weights/best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Export a YOLO model for Raspberry Pi CPU inference.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the trained .pt model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size.")
    parser.add_argument(
        "--format",
        choices=["ncnn", "onnx"],
        default="ncnn",
        help="Export format. NCNN is recommended for Raspberry Pi CPU.",
    )
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset when using --format onnx.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph when using --format onnx.")
    return parser.parse_args()


def load_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Ultralytics is not installed. Install it with: pip install -U ultralytics"
        ) from exc

    return YOLO


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    YOLO = load_yolo()
    model = YOLO(str(model_path))

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
    }

    if args.format == "onnx":
        export_kwargs["opset"] = args.opset
        export_kwargs["simplify"] = args.simplify

    exported_path = model.export(**export_kwargs)
    print(f"Exported Raspberry Pi model: {exported_path}")


if __name__ == "__main__":
    main()
