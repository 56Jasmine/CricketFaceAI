import os
import numpy as np
from PIL import Image
from keras_facenet import FaceNet

TRAIN_DIR = "data/final/train"
OUTPUT_DIR = "outputs"

embedder = FaceNet()

prototypes = {}
print("Creating player prototypes...\n")

for player in sorted(os.listdir(TRAIN_DIR)):
    player_path = os.path.join(TRAIN_DIR, player)

    if not os.path.isdir(player_path):
        continue

    embeddings = []

    for image_name in sorted(os.listdir(player_path)):
        image_path = os.path.join(player_path, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            embedding = embedder.embeddings(
                np.expand_dims(image_array, axis=0)
            )[0]

            # L2 normalize individual embedding
            embedding = embedding / np.linalg.norm(embedding)

            embeddings.append(embedding)

        except Exception as e:
            print(f"Skipping {image_path}: {e}")

    if embeddings:
        # Average all training embeddings for this player
        prototype = np.mean(embeddings, axis=0)

        # Normalize prototype
        prototype = prototype / np.linalg.norm(prototype)

        prototypes[player] = prototype

        print(f"{player}: {len(embeddings)} images → prototype created")

os.makedirs(OUTPUT_DIR, exist_ok=True)

player_names = sorted(prototypes.keys())
prototype_array = np.array([prototypes[player] for player in player_names])

np.save(
    os.path.join(OUTPUT_DIR, "player_prototypes.npy"),
    prototype_array
)

np.save(
    os.path.join(OUTPUT_DIR, "player_names.npy"),
    np.array(player_names)
)

print("\n================================")
print("Prototype database created!")
print("================================")
print(f"Players: {len(player_names)}")
print(f"Prototype shape: {prototype_array.shape}")
print("Saved:")
print("  outputs/player_prototypes.npy")
print("  outputs/player_names.npy")
