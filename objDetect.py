from ultralytics import YOLO

def main():
    # get the trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    #run the model on one of the images
    #conf: confidence threshold which is how confident the model must be to show a detection 
    #      (conf=0.25 means detections greater than or equal to 25% are kept)
    results = model("images/tent.webp", conf=0.02)

    #print results of the model evaluation of the image
    print(results[0].boxes)

    #save the output image with detections (for example if the model outlined a box around an object)
    results[0].save("images/output.jpg")
    return results

if __name__ == "__main__":
    main()
