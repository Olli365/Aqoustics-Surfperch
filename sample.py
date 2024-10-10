import os
import random
import shutil

def copy_random_files(source_folder, destination_folder, num_files=10):
    # Create destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Walk through the directory tree of the source folder
    for root, dirs, files in os.walk(source_folder):
        # Construct the corresponding path in the destination folder
        relative_path = os.path.relpath(root, source_folder)
        destination_subfolder = os.path.join(destination_folder, relative_path)

        # Create the subfolder in the destination folder
        if not os.path.exists(destination_subfolder):
            os.makedirs(destination_subfolder)

        # If there are files in the current directory, select 10 random files and copy them
        if files:
            selected_files = random.sample(files, min(len(files), num_files))
            for file_name in selected_files:
                source_file = os.path.join(root, file_name)
                destination_file = os.path.join(destination_subfolder, file_name)
                shutil.copy2(source_file, destination_file)

    print(f"Finished copying random files to {destination_folder}")

# Example usage:
source_folder_path = '/home/os/aqoustics/Aqoustics-Surfperch/data/V1clusters/'
destination_folder_path = '/home/os/aqoustics/Aqoustics-Surfperch/data/V1/cluster_samples/'
copy_random_files(source_folder_path, destination_folder_path)
