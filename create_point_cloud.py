import subprocess
import os
import sys


def generate_point_cloud(folder_path):
    # Check if the input path is an absolute path, if not, convert it to an absolute path
    if not os.path.isabs(folder_path):
        folder_path = os.path.abspath(folder_path)

    # Path to the images folder (must exist)
    images_folder = os.path.join(folder_path, "images")

    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder does not exist: {images_folder}")

    project_folder_folder_name = os.path.basename(os.path.normpath(os.path.dirname(folder_path)))
    project_folder_name = os.path.basename(os.path.normpath(folder_path))

    # Command to run OpenDroneMap with the images folder and output path
    odm_command = [
        "docker", "run", "-i", "--rm", "-v", f"{os.path.dirname(folder_path)}:/{project_folder_folder_name}",
        "opendronemap/odm", "--project-path", f"/{project_folder_folder_name}", f"{project_folder_name}",
        "--end-with", "odm_filterpoints"
    ]

    try:
        # Run the ODM process and wait for it to complete
        subprocess.run(odm_command, check=True)
        print("3D point cloud generation complete.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during point cloud generation: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python create_point_cloud.py <project_folder_path>")
        sys.exit(1)

    project_folder_path = sys.argv[1]
    generate_point_cloud(project_folder_path)


if __name__ == '__main__':
    main()
