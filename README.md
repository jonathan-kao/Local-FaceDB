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
