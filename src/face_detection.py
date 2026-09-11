import os
import cv2

INPUT_PATH = "data/processed"
OUTPUT_PATH = "data/face_cropped"

IMAGE_SIZE = (224, 224)

MODEL_PATH = "models/face_detector/face_detection_yunet_2023mar.onnx"

CONFIDENCE_THRESHOLD = 0.6

# Create YuNet face detector
face_detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    0.3,
    5000
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

total_images = 0
detected_faces = 0
failed_images = []


for player_name in os.listdir(INPUT_PATH):

    player_folder = os.path.join(INPUT_PATH, player_name)

    if not os.path.isdir(player_folder):
        continue

    output_player_folder = os.path.join(
        OUTPUT_PATH,
        player_name
    )

    os.makedirs(output_player_folder, exist_ok=True)

    print(f"\nProcessing: {player_name}")

    for image_name in os.listdir(player_folder):

        image_path = os.path.join(
            player_folder,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            failed_images.append(image_path)
            continue

        total_images += 1

        height, width = image.shape[:2]

        # YuNet requires the input image size
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)

        if faces is None or len(faces) == 0:
            failed_images.append(image_path)
            continue

        # Select the face with the highest confidence
        best_face = max(
            faces,
            key=lambda face: face[-1]
        )

        confidence = best_face[-1]

        # YuNet bounding box:
        # x, y, width, height
        x, y, w, h = best_face[:4].astype(int)

        # Add padding around face
        padding = int(
            0.20 * max(w, h)
        )

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)

        x2 = min(
            width,
            x + w + padding
        )

        y2 = min(
            height,
            y + h + padding
        )

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            failed_images.append(image_path)
            continue

        # Resize to MobileNetV2 input size
        face_crop = cv2.resize(
            face_crop,
            IMAGE_SIZE
        )

        output_path = os.path.join(
            output_player_folder,
            image_name
        )

        cv2.imwrite(
            output_path,
            face_crop
        )

        detected_faces += 1


print("\n" + "=" * 50)
print("FACE DETECTION RESULTS")
print("=" * 50)

print(f"Total images processed : {total_images}")
print(f"Faces detected         : {detected_faces}")
print(f"Detection failures     : {len(failed_images)}")

if total_images > 0:

    detection_rate = (
        detected_faces / total_images
    ) * 100

    print(
        f"Detection rate         : "
        f"{detection_rate:.2f}%"
    )


if failed_images:

    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/face_detection_failures.txt",
        "w"
    ) as file:

        for image_path in failed_images:
            file.write(image_path + "\n")

    print(
        "\nFailed image list saved to: "
        "outputs/face_detection_failures.txt"
    )

else:

    print("\nNo face detection failures!")