# Local-FaceDB
This repository synergizes the capabilities of YOLOv8 for face detection, the DeepFace library for generating facial embeddings, and SQLite3 for embedding storage, establishing a comprehensive approach to facial recognition.

## System Components and Methodology

### Face Detection with YOLOv8

The system employs YOLOv8, an advanced iteration within the "You Only Look Once" series, for its face detection module. YOLOv8's state-of-the-art architecture enables it to detect faces with remarkable accuracy and speed, which is crucial for the initial phase of the recognition process.

### Embedding Generation Using DeepFace

Upon detection, faces are processed through the DeepFace library, which generates a unique embedding for each face by converting facial features into a numerical representation. This process is essential for creating a distinguishable and quantifiable identity for each detected face.

### Embedding Storage in SQLite3

The generated embeddings are stored in a SQLite3 database. SQLite3 is selected for its lightweight and efficient characteristics, suitable for managing the embeddings and facilitating rapid retrieval and comparison operations, which are pivotal for the recognition phase.

### Recognition Process

The recognition process hinges on calculating the distances between the embeddings of test faces and those stored in the database. By evaluating the similarity between embeddings, the system can identify individuals by matching new face embeddings with existing ones, thus associating the correct name with each recognized face.

## Objective

The primary aim of this project is to devise a facial recognition system that stands on the pillars of accuracy, efficiency, and reliability. By integrating leading-edge technologies and methodologies, this system seeks to offer a solution that not only achieves facial recognition but does so with a high degree of precision and speed.

## Usage Guide

### Preparing the Images

1. **Cropping Images:** Begin by cropping your training and testing images. Utilize the provided Python scripts, `test_crop.py` and `train_crop.py`, to crop the images. This step is crucial for preparing your dataset for the recognition process.

2. **Directory Structure:** Organize your files following this directory tree structure to ensure that the scripts function correctly:

```bash
├── test_crop.py
├── train_crop.py
├── data
│   ├── train
│   ├── test
│   ├── test_224
│   └── train_224
├── csv
│   ├── category.csv
│   └── train.csv
├── yolo
│   └── yolov8n-face.pt
```

Please place the training images in the `data/train` directory and the test images in the `data/test` directory. The `train_crop.py` and `test_crop.py` scripts will output the cropped images to `data/train_224` and `data/test_224`, respectively.

3. **Image Compression:** After cropping the images, compress the `test_224` and `train_224` directories into zip files. These zipped files will be uploaded to Google Drive for further processing in Google Colab.

### Setting Up Google Drive

1. **Upload Zipped Images:** Upload the zipped image directories (`test_224.zip` and `train_224.zip`) to Google Drive. Follow this recommended directory structure in Google Drive to organize your project files:

```bash
├── Colab Notebooks
│   ├── data
│   │   ├── test_224.zip
│   │   └── train_224.zip
│   └── csv
│   │   ├── category.csv
│   │   └── train.csv
│   └── database
```

2. **Running the Notebook:** Execute the `local_facedb.ipynb` notebook in Google Colab. This script processes the images and performs facial recognition tasks.

3. **Output:** The execution of `local_facedb.ipynb` will produce a CSV file containing the predicted labels for each test image. This file will be saved to the `Colab Notebooks/csv` directory in your Google Drive.

By following these steps, you'll be able to prepare your dataset, execute the facial recognition process, and obtain the predicted labels for analysis.


