# CricketFace AI

## Indian Cricketer Identification Using Computer Vision

CricketFace AI is a computer vision system that identifies selected Indian cricketers from an uploaded image.

The system uses face detection, deep face embeddings, and cosine similarity to match an input face with known player representations.

---

## Features

- Face detection using YuNet
- Deep facial feature extraction using FaceNet
- 512-dimensional face embeddings
- Prototype-based player matching
- Cosine similarity for identification
- Top 3 player predictions
- Streamlit web interface
- Support for 10 selected Indian cricketers

---

## Supported Players

1. Hardik Pandya
2. Jasprit Bumrah
3. M.S. Dhoni
4. Rishabh Pant
5. Rohit Sharma
6. Sachin Tendulkar
7. Shubman Gill
8. Suryakumar Yadav
9. Virat Kohli
10. Yuvraj Singh

---

## System Workflow

```text
Input Image
     |
     v
Face Detection
     |
     v
Face Cropping
     |
     v
FaceNet
     |
     v
512-D Face Embedding
     |
     v
Cosine Similarity
     |
     v
Compare with Player Prototypes
     |
     v
Predicted Player
 Methodology
1. Face Detection

The system first detects the face in the uploaded image using the YuNet face detector.

The detected face is cropped and resized before feature extraction.

2. Face Embedding

FaceNet converts the detected face into a 512-dimensional numerical representation called a face embedding.

The embedding represents important facial characteristics in a compact numerical form.

3. Player Prototypes

For each supported player, embeddings from the training images are averaged to create a representative player prototype.

The prototype database contains one embedding representation for each of the 10 supported players.

4. Similarity Matching

The embedding of the uploaded face is compared with all player prototypes using cosine similarity.

The player with the highest similarity score is returned as the predicted identity.

Dataset

The project uses a selected subset of Indian cricketer face images.

The dataset was processed by:

Selecting 10 player classes
Standardizing image dimensions
Detecting and cropping faces
Removing images where a face could not be detected
Creating separate training, validation, and test sets

After face detection:

Total images processed: 250
Faces successfully detected: 242
Face detection rate: 96.8%

The final dataset contains 242 usable face images.

Train / Validation / Test Split

The processed dataset was divided into:

Training images: 164
Validation images: 30
Test images: 48

The final FaceNet evaluation uses only the training images to create player prototypes and evaluates identification on the held-out test images.

Results

The FaceNet prototype-based identification system achieved:

91.67% test accuracy

on the held-out test set of 48 images across 10 supported players.

Correct predictions: 44
Total test images:   48
Accuracy:            91.67%

This result represents closed-set identification among the 10 supported players and should not be interpreted as general face-recognition accuracy.

Technology Stack
Python
TensorFlow
Keras
FaceNet
OpenCV
YuNet
NumPy
Pillow
Streamlit
Project Structure
CricketFaceAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── face_cropped/
│   └── final/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   ├── cricket_player_baseline.keras
│   ├── cricket_player_final.keras
│   └── face_detector/
│       └── face_detection_yunet_2023mar.onnx
│
├── outputs/
│   ├── face_embeddings.npy
│   ├── face_labels.npy
│   ├── face_image_paths.npy
│   ├── player_names.npy
│   ├── player_prototypes.npy
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── face_detection_failures.txt
│   └── training_history.json
│
└── src/
    ├── preprocess.py
    ├── face_detection.py
    ├── create_split.py
    ├── create_embeddings.py
    ├── create_prototypes.py
    ├── predict.py
    ├── evaluate_facenet.py
    ├── evaluate_facenet_report.py
    ├── data_loader.py
    ├── model.py
    ├── train.py
    └── evaluate.py
**How to Run**
1. Create and activate the environment
   conda activate cricketface
2. Install dependencies
   pip install -r requirements.txt
3. Run the Streamlit application
   streamlit run app.py
   The application will open in the browser

**Future Improvements**
Increase the number of training images per player
Add more Indian cricketers
Improve performance on difficult lighting and side-profile images
Add multi-face detection and identification
Add an unknown-person rejection threshold
Improve the user interface
Deploy the application online
