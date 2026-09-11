import os
import shutil
import random

SOURCE_DIR = "data/face_cropped"
OUTPUT_DIR = "data/final"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

# Create output directories
for split in ["train", "validation", "test"]:
    os.makedirs(
        os.path.join(OUTPUT_DIR, split),
        exist_ok=True
    )

total_images = 0

for player_name in os.listdir(SOURCE_DIR):

    player_folder = os.path.join(
        SOURCE_DIR,
        player_name
    )

    if not os.path.isdir(player_folder):
        continue

    images = [
        image for image in os.listdir(player_folder)
        if image.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[
        train_count:train_count + val_count
    ]
    test_images = images[
        train_count + val_count:
    ]

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    print(f"\n{player_name}")
    print(f"Total      : {total}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")

    for split, split_images in splits.items():

        destination_folder = os.path.join(
            OUTPUT_DIR,
            split,
            player_name
        )

        os.makedirs(
            destination_folder,
            exist_ok=True
        )

        for image_name in split_images:

            source_path = os.path.join(
                player_folder,
                image_name
            )

            destination_path = os.path.join(
                destination_folder,
                image_name
            )

            shutil.copy2(
                source_path,
                destination_path
            )

            total_images += 1


print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)

print(f"Total images copied: {total_images}")
print(f"Random seed: {SEED}")