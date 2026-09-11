import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

NUM_CLASSES = 10
IMAGE_SIZE = (224, 224)


def build_model():

    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Initially freeze the pretrained network
    base_model.trainable = False

    model = models.Sequential([

        layers.Input(
            shape=IMAGE_SIZE + (3,)
        ),

        # Stronger augmentation for the small dataset
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            height_factor=(-0.10, 0.15),
            width_factor=(-0.10, 0.15)
        ),

        layers.RandomTranslation(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomContrast(
            0.15
        ),

        # MobileNetV2 preprocessing
        layers.Rescaling(
            1.0 / 127.5,
            offset=-1
        ),

        # Pretrained feature extractor
        base_model,

        layers.GlobalAveragePooling2D(),

        # Helps reduce overfitting
        layers.Dropout(
            0.5
        ),

        # Classification layer
        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    ])

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()