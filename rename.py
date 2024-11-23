import os

# Tentukan path folder dataset_cabai
base_folder = 'dataset_cabai'

# Daftar folder yang ingin di-rename
folders = ['busuk', 'sehat']

# Loop melalui setiap folder
for folder in folders:
    folder_path = os.path.join(base_folder, folder)
    
    # Pastikan folder ada
    if os.path.exists(folder_path):
        # Dapatkan daftar semua file dalam folder
        files = os.listdir(folder_path)

        # Loop melalui setiap file dan rename
        for index, filename in enumerate(files):
            # Tentukan path file lama
            old_file = os.path.join(folder_path, filename)
            
            # Tentukan nama file baru (misalnya, menambahkan prefix sesuai folder)
            new_file = os.path.join(folder_path, f'{folder}_{index + 1}.jpg') 

            # Rename file
            os.rename(old_file, new_file)

        print(f"Semua file di folder '{folder}' telah di-rename.")
    else:
        print(f"Folder '{folder}' tidak ditemukan di '{base_folder}'.")