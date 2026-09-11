import os
import json
import tensorflow as tf

from data_loader import load_datasets
from model import build_model


# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 15

BASE_MODEL_PATH = "models/cricket_player_baseline.keras"
FINAL_MODEL_PATH = "models/cricket_player_final.keras"

HISTORY_PATH = "outputs/training_history.json"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

train_dataset, validation_dataset, test_dataset, class_names = load_datasets()

print("\n" + "=" * 60)
print("PLAYER CLASSES")
print("=" * 60)

for i, player in enumerate(class_names):
    print(f"{i}: {player}")


# ============================================================
# BUILD MODEL
# ============================================================

model = build_model()

print("\n" + "=" * 60)
print("MODEL CREATED")
print("=" * 60)

model.summary()


# ============================================================
# INITIAL TRAINING
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


print("\n" + "=" * 60)
print("STARTING INITIAL TRAINING")
print("=" * 60)


checkpoint_initial = tf.keras.callbacks.ModelCheckpoint(
    BASE_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping_initial = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr_initial = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


history_initial = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS,
    callbacks=[
        checkpoint_initial,
        early_stopping_initial,
        reduce_lr_initial
    ]
)


print("\nBaseline training completed.")


# ============================================================
# FINE-TUNING
# ============================================================

print("\n" + "=" * 60)
print("STARTING FINE-TUNING")
print("=" * 60)


# Find MobileNetV2 inside the Sequential model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "mobilenetv2" in layer.name.lower():
            base_model = layer
            break


if base_model is None:
    raise ValueError(
        "MobileNetV2 base model not found."
    )


# Unfreeze MobileNetV2
base_model.trainable = True


# Freeze earlier layers
for layer in base_model.layers[:-30]:
    layer.trainable = False


# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False


print(
    "\nFine-tuning the last 30 MobileNetV2 layers."
)


# Very small learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint_final = tf.keras.callbacks.ModelCheckpoint(
    FINAL_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping_final = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr_final = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[
        checkpoint_final,
        early_stopping_final,
        reduce_lr_final
    ]
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(FINAL_MODEL_PATH)

print("\n" + "=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)

print(
    f"Model path: {FINAL_MODEL_PATH}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history = {
    "initial_training": {
        key: [float(value) for value in values]
        for key, values in history_initial.history.items()
    },
    "fine_tuning": {
        key: [float(value) for value in values]
        for key, values in history_fine.history.items()
    }
}


with open(HISTORY_PATH, "w") as file:
    json.dump(
        history,
        file,
        indent=4
    )


print(
    f"Training history saved: {HISTORY_PATH}"
)
# Load the best baseline model before fine-tuning
model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)

print("\nBest baseline model loaded for fine-tuning.")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)