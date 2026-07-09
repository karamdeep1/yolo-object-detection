# Yolo Object Detection

## Required installations
1. install ultralytics through pip
```
pip install -U ultralytics
```

2. install numpy
```
pip install numpy
```

3. install pytorch
```
pip3 install torch torchvision
```
Note that this will install pytorch to run off the CPU. If you have a NVIDIA GPU that is compatible with PyTorch, visit the website and download the correct CUDA version according to your GPU: 
https://pytorch.org/get-started/locally/


## Adding Datasets
### Personal Datasets
To add datasets to train the model on, add the dataset to the [dataset directory](datasets) and then in **objDetectTrain.py** change the location of the current dataset to the one you added. After that you can run the **objDetectTrain.py** script to train your model.

### Yaml
A yaml file is necessary for datasets for the model to train on it. It is easy to make the yaml file. A yaml file is provided in the [peopleAndTents](https://drive.google.com/drive/folders/1_umBWMeEei_BR6PmaZI4nMzlpph0j3SL?usp=drive_link) dataset. You can use this as a reference to make your own yaml file.

- path: this is the directory path where your dataset is
- train: this is the directory where your training data is in your dataset directory
- val: this is the directory where your validating data is in your dataset directory
- test: this is the directory where your testing data is in your dataset directory

Notice how there is a section called **names**. This is an important section because it has the class ID's that your model should be detecting. Basically your dataset should have a 0 or 1 or some number at the beginning of each label file. That number is the class ID that corresponds to the ID specified in the yaml file. For example **Tent** has a ID of 1 so all the labels for the Tent data in the peopleAndTents dataset starts with a 1. All the labels for the People data starts with a 0.

### Merging Multiple Datasets
There is a script called **convertClassID.py** that can help you update all the class ID's in a dataset that way you do not have to manually go through and change all the ID's one by one. You can edit the class ID you want to change by editing the numbers in the following if statement in the script:
```
if parts[0] == "0":
  parts[0] = "1"
```

This is useful because most datasets have their class ID's as 0 or some other number. So if you decide to make your own yaml file and need to make sure that all your datasets match the class ID's you specified in the yaml file, you can utilize this script to change the class ID's of your datasets to match your yaml file. After that you can put all the correlated data into the same directories.

For example, the [peopleAndTents](https://drive.google.com/drive/folders/1_umBWMeEei_BR6PmaZI4nMzlpph0j3SL?usp=drive_link) dataset was the result of a merged dataset. There was a people dataset and a tent dataset. Both datasets had a class ID of 0 so I utilized the script to change the tents dataset class ID to 1 to match the **peopleAndTents.yaml** file and then I moved the contents of the both datasets into 1 big dataset.


## Training The Model
### Utilizing peopleAndTents Dataset
There is a dataset available for download called [peopleAndTents](https://drive.google.com/drive/folders/1_umBWMeEei_BR6PmaZI4nMzlpph0j3SL?usp=drive_link). After downloading this and extracting (I recommend extracting with 7-Zip) make sure you add it to the datasets directory. The datasets directory should have this structure once you add it:
```
datasets/
  peopleAndTents/
    images/
    labels/
    peopleAndTents.yaml
```


### Running Training Script
Make sure there is a dataset that the model can train on. There currently already is a dataset for the model to train on called [peopleAndTents](https://drive.google.com/drive/folders/1_umBWMeEei_BR6PmaZI4nMzlpph0j3SL?usp=drive_link). To train the model run the **objDetectTrain.py** script like so:
```
python ./objDetectTrain.py 
```

### Train Directory
Currently there is a [runs](runs) directory which contains a **yolo26m.pt** model that was trained on the [peopleAndTents](https://drive.google.com/drive/folders/1_umBWMeEei_BR6PmaZI4nMzlpph0j3SL?usp=drive_link) dataset. When you train your model, you will see a train directory created in the runs directory. If you train multiple times, there will be multiple train directories created with incremental names (E.g. train, train1, train2, etc).

Inside this train directory, you will see a **weights** directory which has the best model created from training and the most recent model created from training. You can use whichever model you wish to use object detection on images in the **objDetect.py** script. Just make sure you change the model in the script.

## Running The Model
`objDetect.py` runs detection from the command line. By default it uses:

- model: `runs/detect/train3/weights/best.pt`
- source image: `images/canopyTentPeople.webp`
- classes:
  - `0`: Person
  - `1`: Tent

To test the default image, run:
```
python objDetect.py
```

To test a different image, pass `--source`:
```
python objDetect.py --source images/canopyTent.jpg
```

To run detection on a whole folder of images:
```
python objDetect.py --source images
```

To save annotated prediction images, add `--save`:
```
python objDetect.py --source images/canopyTent.jpg --save
```

Saved predictions are written by Ultralytics into a `runs/detect/predict...` directory.

### Tent-Only Detection
If you only care about canopy tents, use `--classes 1`. This ignores people and only returns the `Tent` class:
```
python objDetect.py --source images/canopyTentPeople.webp --classes 1
```

### Fast Airborne Inference
For airborne use, use `--fast`. This turns on speed-focused settings like FP16 inference, disables saving, and limits the maximum detections per frame:
```
python objDetect.py --fast --classes 1
```

To benchmark FPS on the default image:
```
python objDetect.py --fast --benchmark --runs 100 --classes 1
```

To test different image sizes for the FPS/accuracy tradeoff:
```
python objDetect.py --fast --benchmark --runs 100 --classes 1 --imgsz 640
python objDetect.py --fast --benchmark --runs 100 --classes 1 --imgsz 512
python objDetect.py --fast --benchmark --runs 100 --classes 1 --imgsz 416
```

Lower `imgsz` usually gives more FPS, but can miss small tents from altitude. Test with real flight imagery before choosing the final value.

### Video Or Camera Input
For a flight video, use `--stream` so frames are processed one at a time:
```
python objDetect.py --fast --source path\to\flight_video.mp4 --stream --classes 1
```

To skip frames for more throughput, use `--vid-stride`. For example, this runs detection on every second frame:
```
python objDetect.py --fast --source path\to\flight_video.mp4 --stream --classes 1 --vid-stride 2
```

For a webcam or camera device:
```
python objDetect.py --fast --source 0 --stream --classes 1
```

### TensorRT For Maximum NVIDIA GPU FPS
If you have an NVIDIA GPU, TensorRT is usually faster than the `.pt` or `.onnx` model.

Export the model:
```
python exportModel.py
```

Then run the exported engine:
```
python objDetect.py --model runs/detect/train3/weights/best.engine --fast --benchmark --runs 100 --classes 1
```

If `runs/detect/train3/weights/best.engine` exists, `objDetect.py` will automatically use it when you run the default model path.

### Useful Options
- `--model`: path to a `.pt`, `.onnx`, or `.engine` model
- `--source`: image, folder, video, camera index, or stream URL
- `--imgsz`: inference image size, such as `640`, `512`, or `416`
- `--conf`: confidence threshold
- `--iou`: NMS IoU threshold
- `--device`: device to use, such as `0` for GPU or `cpu`
- `--half`: use FP16 inference when supported
- `--fast`: speed-focused airborne defaults
- `--save`: save annotated prediction images
- `--stream`: process video/camera sources frame by frame
- `--vid-stride`: run inference on every Nth video frame
- `--max-det`: maximum detections per image/frame
- `--classes`: comma-separated class IDs, such as `1` for tents only
- `--benchmark`: run repeated inference and print average FPS
- `--json`: print formatted detection JSON
