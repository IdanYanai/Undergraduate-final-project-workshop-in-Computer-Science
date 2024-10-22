import readline
import os
import glob
from extract_images import extract_frames
from create_point_cloud import generate_point_cloud
from merge_clouds import merge_clouds


def complete_path(text, state):
    if '~' in text:
        text = os.path.expanduser(text)
    results = glob.glob(text + '*') + [None]
    return results[state]


def main():
    help_message = (
        "========================================\n"
        "Hello, welcome to our Photogrammetry tool!\n"
        "Authors: Idan Yanai & Ariel Dvori\n\n"
        
        "Commands:\n"
        "1. Extract (extract images from a video)\n"
        "2. Create (create a point cloud from images)\n"
        "3. Merge (merge two point clouds)\n"
        "========================================"
    )
    print(help_message)

    readline.parse_and_bind("tab: complete")
    readline.set_completer(complete_path)
    readline.parse_and_bind("set editing-mode vi")

    while True:
        try:
            command = input("-->").strip()
        except KeyboardInterrupt:
            print()
            break

        if command.lower() in ["quit", "exit"]:
            break
        if command in ["-h", "--help"]:
            print(help_message)
            continue

        try:
            if command.lower() == "extract":
                args = input("Please Enter: <video_path> <output_folder> <number_of_images>\n--(Extract)-->").strip()
                if len(args.split()) != 3:
                    print("Bad parameters")
                    continue
                video_path, output_folder, k = args.split()
                extract_frames(video_path, output_folder, int(k))

            elif command.lower() == "create":
                folder_path = input("Please enter: <project_folder_path>\n--(Create)-->").strip()
                generate_point_cloud(folder_path)

            elif command.lower() == "merge":
                args = input("Please enter: <file1> <file2> <output_file> <icp>(0/1) (<voxel_size>=0.05)\n--(Merge)-->").strip()
                if len(args.split()) == 5:
                    file1, file2, output_file, icp, voxel_size = args.split()
                    merge_clouds(file1, file2, output_file, int(icp), float(voxel_size))
                elif len(args.split()) == 4:
                    file1, file2, output_file, icp = args.split()
                    merge_clouds(file1, file2, output_file, int(icp))
                else:
                    print("Bad parameters")
                    continue
            else:
                print("Not a valid command | use -h for help")
        except KeyboardInterrupt:
            print("\nBack to main commands...")
            continue
        except RuntimeError as e:
            print(e)
            continue


if __name__ == '__main__':
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind("tab: complete")
    main()
