# Could see in pics moth 5 and 6 were mixed up

import os
directory = r'D:\mars_global_acoustic_study\maldives_acoustics\R1_M5'

current_site = 'D2'
new_site = 'R1'

def rename_specific_files(directory: str) -> None:
    for root, _, files in os.walk(directory):
        for file in files:
            if file[4:6] == current_site:
                new_name = file[:4] + new_site + file[6:]
                os.rename(os.path.join(root, file), os.path.join(root, new_name))
                print(f'Renamed: {file} to {new_name}')

rename_specific_files(directory)
