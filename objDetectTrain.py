from ultralytics import YOLO

def main():
    # Load a COCO-pretrained YOLO26m model
    model = YOLO("models/yolo26m.pt")
    
    #data: the data to train on
    #epochs: the amount of runs the model goes through the data
    #batch: the amount of images processed in one run (helps not running out of memory)
    #imgsz: image size
    #device: 0 means using first gpu, 1 would be the 2nd gpu if you have one, "cpu" would mean using the cpu
    #workers: 0 means everything runs in main thread which is safe to avoid multiprocessing error (windows issue), 8 means 8 parallel loaders which is faster loading
    results = model.train(data="datasets/peopleAndTents/peopleAndTents.yaml", 
                          epochs=3,
                          batch=2,
                          imgsz=640,
                          device=0,
                          workers=0)
    return results

if __name__ == "__main__":
    main()
