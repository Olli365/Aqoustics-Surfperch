import os

def rename_audio_files(folder_path):
    total_files_found = 0  # Counter for total files found

    # Loop through each folder in the main folder
    for root, dirs, files in os.walk(folder_path):
        if not files:
            print(f"No files found in {root}")
        else:
            print(f"Found {len(files)} files in {root}")
        
        for file in files:
            # Check if the file is an audio file (adjust extensions if necessary)
            if file.endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg')):
                total_files_found += 1

                # Split the file name into parts
                filename, ext = os.path.splitext(file)

                # Only consider files that have '_clip_' in their name
                if '_clip_' in filename:
                    # Split the part before '_clip_' and after
                    before_clip, after_clip = filename.split('_clip_', 1)
                    
                    # Split based on underscores for the part before '_clip_'
                    parts = before_clip.split('_')

                    if len(parts) >= 2:
                        # First part (Healthy, Restored, Degraded) should be shortened
                        if parts[0] == 'Healthy':
                            new_category = 'H'
                        elif parts[0] == 'Restored':
                            new_category = 'R'
                        elif parts[0] == 'Degraded':
                            new_category = 'D'
                        else:
                            print(f"Unknown category for file {file}")
                            continue
                        
                        # Second part (Moth32) should be reduced to just the number
                        if parts[1].startswith('Moth'):
                            new_number = parts[1].replace('Moth', '')

                            # New file name format: H32_clip_...
                            new_filename = f"{new_category}{new_number}_clip_{after_clip}{ext}"

                            # Full path to the old and new file
                            old_file_path = os.path.join(root, file)
                            new_file_path = os.path.join(root, new_filename)

                            # Rename the file
                            os.rename(old_file_path, new_file_path)
                            print(f"Renamed: {old_file_path} -> {new_file_path}")
                    else:
                        print(f"Skipping file with unexpected name format: {file}")

    # After all files are processed, print the total count
    print(f"Total audio files found and processed: {total_files_found}")



folder_path = "/mnt/d/Aqoustics/BEN/Australia_Clusters/"
rename_audio_files(folder_path)
