import os
from PIL import Image


DATA_PATH = "data/processed"
IMAGE_SIZE = (224, 224)


def preprocess_images():
    """
    Loads images from the processed dataset,
    converts them to RGB, resizes them,
    and removes corrupted images.
    """

    total_images = 0
    valid_images = 0
    corrupted_images = 0

    for player in os.listdir(DATA_PATH):

        player_path = os.path.join(DATA_PATH, player)

        if not os.path.isdir(player_path):
            continue

        print(f"\nProcessing: {player}")

        for image_name in os.listdir(player_path):

            image_path = os.path.join(
                player_path,
                image_name
            )

            total_images += 1

            try:

                with Image.open(image_path) as image:

                    # Convert image to RGB
                    image = image.convert("RGB")

                    # Resize image
                    image = image.resize(IMAGE_SIZE)

                    # Save processed image
                    image.save(image_path)

                    valid_images += 1

            except Exception as e:

                print(
                    f"Corrupted image: {image_name}"
                )

                os.remove(image_path)

                corrupted_images += 1


    print("\n" + "=" * 40)

    print("PREPROCESSING COMPLETE")

    print(f"Total images: {total_images}")

    print(f"Valid images: {valid_images}")

    print(f"Corrupted images removed: {corrupted_images}")

    print("=" * 40)


if __name__ == "__main__":

    preprocess_images()