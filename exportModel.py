from ultralytics import YOLO

def main():

    # get the trained model
    model = YOLO("runs/detect/train3/weights/best.pt")

    model.export(
        format="engine",
        half=True
    )

if __name__ == "__main__":
    main()