import os
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DIR = "data/final/train"
TEST_DIR = "data/final/test"

embedder = FaceNet()


def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    embedding = embedder.embeddings(
        np.expand_dims(image_array, axis=0)
    )[0]

    return embedding / np.linalg.norm(embedding)


# -----------------------------
# Create prototypes from train
# -----------------------------

prototypes = {}

for player in sorted(os.listdir(TRAIN_DIR)):

    player_path = os.path.join(TRAIN_DIR, player)

    if not os.path.isdir(player_path):
        continue

    embeddings = []

    for image_name in os.listdir(player_path):

        image_path = os.path.join(player_path, image_name)

        try:
            embeddings.append(get_embedding(image_path))
        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    prototype = np.mean(embeddings, axis=0)
    prototype = prototype / np.linalg.norm(prototype)

    prototypes[player] = prototype


# -----------------------------
# Predict test images
# -----------------------------

y_true = []
y_pred = []

for player in sorted(os.listdir(TEST_DIR)):

    player_path = os.path.join(TEST_DIR, player)

    if not os.path.isdir(player_path):
        continue

    for image_name in sorted(os.listdir(player_path)):

        image_path = os.path.join(player_path, image_name)

        try:
            embedding = get_embedding(image_path)

            scores = {}

            for candidate, prototype in prototypes.items():
                scores[candidate] = np.dot(
                    embedding,
                    prototype
                )

            predicted_player = max(scores, key=scores.get)

            y_true.append(player)
            y_pred.append(predicted_player)

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


# -----------------------------
# Evaluation
# -----------------------------

players = sorted(prototypes.keys())

print("\n==============================")
print("FaceNet Classification Report")
print("==============================\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=players,
        target_names=players,
        zero_division=0
    )
)

print("==============================")
print("Confusion Matrix")
print("==============================\n")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=players
)

print("Class order:")
for i, player in enumerate(players):
    print(f"{i}: {player}")

print("\nMatrix:")
print(cm)