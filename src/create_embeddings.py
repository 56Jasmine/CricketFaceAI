import os
import numpy as np
from PIL import Image
from keras_facenet import FaceNet

SOURCE_DIR = "data/face_cropped"
OUTPUT_DIR = "outputs"

embedder = FaceNet()

embeddings = []
labels = []
image_paths = []

print("Generating FaceNet embeddings...\n")

for player in sorted(os.listdir(SOURCE_DIR)):

    player_path = os.path.join(SOURCE_DIR, player)

    if not os.path.isdir(player_path):
        continue

    print(f"Processing: {player}")

    for image_name in sorted(os.listdir(player_path)):

        image_path = os.path.join(player_path, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image)

            # FaceNet expects a batch of images
            embedding = embedder.embeddings(
                np.expand_dims(image_array, axis=0)
            )[0]

            embeddings.append(embedding)
            labels.append(player)
            image_paths.append(image_path)

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(
    os.path.join(OUTPUT_DIR, "face_embeddings.npy"),
    np.array(embeddings)
)

np.save(
    os.path.join(OUTPUT_DIR, "face_labels.npy"),
    np.array(labels)
)

np.save(
    os.path.join(OUTPUT_DIR, "face_image_paths.npy"),
    np.array(image_paths)
)

print("\n================================")
print("Embedding generation completed!")
print("================================")
print(f"Total embeddings: {len(embeddings)}")
print(f"Embedding shape: {np.array(embeddings).shape}")
print(f"Labels saved: outputs/face_labels.npy")
print(f"Embeddings saved: outputs/face_embeddings.npy")