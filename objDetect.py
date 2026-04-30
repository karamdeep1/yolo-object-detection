from ultralytics import YOLO
import time

def main():
    # get the trained model
    model = YOLO("runs/detect/train3/weights/best.pt")

    results = model.predict(
        source="images/canopyTentPeople.webp",
        imgsz=640,
        conf=0.15,
        device=0,
        half=True,
        save=False,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_id": cls_id,
            "class_name": results[0].names[cls_id],
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    print(detections)

    return detections

def optimizedModel():
    model = YOLO("runs/detect/train3/weights/best.onnx")

    #warm up
    for _ in range(3):
        results = model.predict(
            source="images/canopyTent.jpg",
            imgsz=640,
            conf=0.15,
            device=0,
            save=True,
            verbose=False
        )

    #actual runs
    runs = 20

    start = time.time()
    for _ in range(runs):
        results = model.predict(
            source="images/canopyTent.jpg",
            imgsz=640,
            conf=0.15,
            device=0,
            save=True,
            verbose=False
        )
    end = time.time()


    avg_time = (end - start) / runs
    fps = 1 / avg_time

    print(f"Average time per image: {avg_time:.4f} s")
    print(f"Average FPS: {fps:.2f}")

    print(results[0].boxes)

if __name__ == "__main__":
    optimizedModel()
