import os
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

TRAIN_DIR = "data/final/train"
TEST_DIR = "data/final/test"

embedder = FaceNet()


def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    embedding = embedder.embeddings(
        np.expand_dims(image_array, axis=0)
    )[0]

    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# --------------------------------
# 1. Create prototypes from TRAIN
# --------------------------------

print("Creating player prototypes...\n")

prototypes = {}

for player in sorted(os.listdir(TRAIN_DIR)):

    player_path = os.path.join(TRAIN_DIR, player)

    if not os.path.isdir(player_path):
        continue

    player_embeddings = []

    for image_name in os.listdir(player_path):

        image_path = os.path.join(player_path, image_name)

        try:
            embedding = get_embedding(image_path)
            player_embeddings.append(embedding)

        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    prototype = np.mean(player_embeddings, axis=0)
    prototype = prototype / np.linalg.norm(prototype)

    prototypes[player] = prototype

    print(f"{player}: {len(player_embeddings)} training images")


# --------------------------------
# 2. Test on completely unseen data
# --------------------------------

print("\nTesting on unseen test images...\n")

correct = 0
total = 0

for player in sorted(os.listdir(TEST_DIR)):

    player_path = os.path.join(TEST_DIR, player)

    if not os.path.isdir(player_path):
        continue

    for image_name in os.listdir(player_path):

        image_path = os.path.join(player_path, image_name)

        try:
            embedding = get_embedding(image_path)

            scores = {}

            for candidate, prototype in prototypes.items():

                score = cosine_similarity(
                    embedding.reshape(1, -1),
                    prototype.reshape(1, -1)
                )[0][0]

                scores[candidate] = score

            predicted_player = max(scores, key=scores.get)

            if predicted_player == player:
                correct += 1

            total += 1

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


accuracy = correct / total


print("==============================")
print("FaceNet Held-Out Test")
print("==============================")
print(f"Correct predictions: {correct}/{total}")
print(f"Test accuracy: {accuracy:.2%}")