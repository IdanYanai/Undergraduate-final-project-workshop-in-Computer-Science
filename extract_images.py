import cv2
import os
import sys
from PIL import Image
import imagehash


def compute_hash(image_path):
    image = Image.open(image_path)
    hash = imagehash.average_hash(image)
    return hash


def image_similarity(hash1, hash2):
    return 1 - (hash1 - hash2) / len(hash1.hash) ** 2


def delete_similar_images(folder_path, threshold=0.9):
    image_files = os.listdir(folder_path)

    for i, image_file1 in enumerate(image_files):
        try:
            image_path1 = os.path.join(folder_path, image_file1)
            hash1 = compute_hash(image_path1)

            for image_file2 in image_files[i + 1:]:
                image_path2 = os.path.join(folder_path, image_file2)
                hash2 = compute_hash(image_path2)

                similarity = image_similarity(hash1, hash2)

                if similarity > threshold:
                    print(f"Similarity between {image_file1} and {image_file2}: {similarity}")
                    # Delete one of the images (you may want to modify this part)
                    os.remove(image_path2)
                    print(f"{image_file2} deleted.")

        except FileNotFoundError:
            continue


def extract_frames(video_path, output_folder, k):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate intervals
    interval = total_frames // (k + 1)

    # Extract frames
    frames = []
    for i in range(1, k + 1):
        frame_number = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print(f"Error: Could not read frame {frame_number}")
            break

    # Save extracted frames
    for i, frame in enumerate(frames):
        frame_name = os.path.join(output_folder, f"frame_{i + 1}.jpg")
        cv2.imwrite(frame_name, frame)
        print(f"Frame {i + 1} saved as {frame_name}")

    # Release video capture
    cap.release()

    # Delete similar images
    delete_similar_images(output_folder)


def main():
    if len(sys.argv) != 4:
        print("Usage: python extract_images.py <video_path> <output_folder> <K>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    k = int(sys.argv[3])

    # Extract frames
    extract_frames(video_path, output_folder, k)


if __name__ == '__main__':
    main()
