import tensorflow as tf

# Final dataset paths
TRAIN_PATH = "data/final/train"
VAL_PATH = "data/final/validation"
TEST_PATH = "data/final/test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42


def load_datasets():

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_dataset.class_names

    # Improve input pipeline performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    validation_dataset = validation_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    test_dataset = test_dataset.prefetch(
        buffer_size=AUTOTUNE
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    )


if __name__ == "__main__":

    train_ds, val_ds, test_ds, classes = load_datasets()

    print("\nClasses:")

    for i, player in enumerate(classes):
        print(f"{i}: {player}")