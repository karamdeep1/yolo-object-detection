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
To test the model on an image you can run the **objDetect.py** script like so:
```
python ./objDetect.py
```

To test the model on different images you can just change the image directory in **objDetect.py**. You should see the output in **images/output.jpg**. Results will also print to the terminal as well