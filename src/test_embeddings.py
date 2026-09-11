import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = "outputs/face_embeddings.npy"
LABELS_PATH = "outputs/face_labels.npy"

embeddings = np.load(EMBEDDINGS_PATH)
labels = np.load(LABELS_PATH)

# Create one average embedding (prototype) for each player
players = sorted(set(labels))
prototypes = {}

for player in players:
    player_embeddings = embeddings[labels == player]

    # Average embedding for this player
    prototype = player_embeddings.mean(axis=0)

    # Normalize prototype
    prototype = prototype / np.linalg.norm(prototype)

    prototypes[player] = prototype

# Test every image against all player prototypes
correct = 0
total = len(embeddings)

for embedding, actual_player in zip(embeddings, labels):

    # Normalize image embedding
    embedding = embedding / np.linalg.norm(embedding)

    scores = {}

    for player, prototype in prototypes.items():
        score = cosine_similarity(
            embedding.reshape(1, -1),
            prototype.reshape(1, -1)
        )[0][0]

        scores[player] = score

    predicted_player = max(scores, key=scores.get)

    if predicted_player == actual_player:
        correct += 1

accuracy = correct / total

print("\n==============================")
print("FaceNet Prototype Test")
print("==============================")
print(f"Correct predictions: {correct}/{total}")
print(f"Accuracy: {accuracy:.2%}")