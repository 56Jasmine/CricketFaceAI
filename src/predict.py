import os
import sys
import cv2
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_DIR = "data/final/train"
YUNET_MODEL = "models/face_detector/face_detection_yunet_2023mar.onnx"

embedder = FaceNet()


def detect_and_crop_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read the input image.")

    height, width = image.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (width, height),
        0.6,
        0.3,
        5000
    )

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected.")

    # Select the highest-confidence detected face
    face = max(faces, key=lambda x: x[-1])

    x, y, w, h = face[:4].astype(int)

    # Add 20% padding around the face
    padding = int(0.20 * max(w, h))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise ValueError("Invalid face crop.")

    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = cv2.resize(face_crop, (160, 160))

    return face_crop


def get_embedding(face_image):
    embedding = embedder.embeddings(
        np.expand_dims(face_image, axis=0)
    )[0]

    return embedding / np.linalg.norm(embedding)


# --------------------------------
# Create player prototypes
# --------------------------------

print("Creating player database...\n")

prototypes = {}

for player in sorted(os.listdir(TRAIN_DIR)):

    player_path = os.path.join(TRAIN_DIR, player)

    if not os.path.isdir(player_path):
        continue

    player_embeddings = []

    for image_name in os.listdir(player_path):

        image_path = os.path.join(player_path, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            embedding = get_embedding(image_array)
            player_embeddings.append(embedding)

        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    prototype = np.mean(player_embeddings, axis=0)
    prototype = prototype / np.linalg.norm(prototype)

    prototypes[player] = prototype


# --------------------------------
# Input image
# --------------------------------

if len(sys.argv) < 2:
    print("\nUsage:")
    print("python src/predict.py path/to/image.jpg")
    sys.exit()

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"\nImage not found: {image_path}")
    sys.exit()


# --------------------------------
# Detect face
# --------------------------------

print("\nDetecting face...")

try:
    face = detect_and_crop_face(image_path)
except Exception as e:
    print(f"\nError: {e}")
    sys.exit()


# --------------------------------
# Generate embedding
# --------------------------------

print("Generating face embedding...")

embedding = get_embedding(face)


# --------------------------------
# Compare with player prototypes
# --------------------------------

scores = {}

for player, prototype in prototypes.items():

    score = cosine_similarity(
        embedding.reshape(1, -1),
        prototype.reshape(1, -1)
    )[0][0]

    scores[player] = score


ranked = sorted(
    scores.items(),
    key=lambda x: x[1],
    reverse=True
)

best_player, best_score = ranked[0]


# --------------------------------
# Display result
# --------------------------------

print("\n==============================")
print("CricketFace AI Prediction")
print("==============================")

print(f"\nPredicted player: {best_player}")
print(f"Similarity score: {best_score:.4f}")

print("\nTop 3 predictions:")

for player, score in ranked[:3]:
    print(f"{player}: {score:.4f}")