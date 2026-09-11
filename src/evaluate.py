import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from data_loader import load_datasets


MODEL_PATH = "models/cricket_player_final.keras"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

_, _, test_dataset, class_names = load_datasets()

print("\n" + "=" * 60)
print("LOADING FINAL MODEL")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model loaded: {MODEL_PATH}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING TEST DATA")
print("=" * 60)

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("TEST ACCURACY")
print("=" * 60)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


np.savetxt(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    ),
    cm,
    delimiter=",",
    fmt="%d"
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    "Classification report saved to:"
    " outputs/classification_report.txt"
)

print(
    "Confusion matrix saved to:"
    " outputs/confusion_matrix.csv"
)