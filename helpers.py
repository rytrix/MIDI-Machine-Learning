import os
import shutil
import json
import random
import numpy as np

def load_songs_random(tokenizer, paths, n):
    train_data = []
    for i in random.sample(paths, n):
        train_data.extend(tokenizer.encode(i).ids)
    return train_data

def load_songs(songs_array, tokenizer, paths):
    train_data = []
    for i in songs_array:
        train_data.extend(tokenizer.encode(paths[i]).ids)
    return train_data

def setup_timestep_data(train_data, timesteps):
    x = []
    y = []

    total_groups = int(len(train_data) / timesteps) - 1
    total_range = timesteps * total_groups

    for i in range(total_range):
        x.append(train_data[i:i+timesteps])
        y.append(train_data[i+timesteps])

    x = np.array(x)
    y = np.array(y)

    return x, y

def create_random_midi_folder(folders = [(5, "folder_name")], source_dir = "midis"):
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    midi_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".mid")]

    if not midi_files:
        print(f"No .mid files found in '{source_dir}'.")
        return

    for num_files, new_folder_name in folders:
        if num_files > len(midi_files):
            print(f"Requested {num_files} files, but only {len(midi_files)} found. Using all available files.")
            num_files = len(midi_files)
            if num_files == 0:
                print("No files left to copy. Exiting.")
                return
        
        if os.path.exists(new_folder_name):
            print(f"Folder '{new_folder_name}' already exists. Stopping.")
            return

        selected_files = random.sample(midi_files, num_files)

        for file in selected_files:
            midi_files.remove(file)

        try:
            os.makedirs(new_folder_name, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory '{new_folder_name}': {e}")
            return

        for file in selected_files:
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(new_folder_name, file)
            try:
                shutil.copy2(source_path, destination_path)
            except OSError as e:
                print(f"Error copying file '{file}': {e}")

        print(f"Successfully created folder '{new_folder_name}' with {num_files} random .mid files.")

def create_model_folder(output_directory_name, model_name, tokenizer_name, timesteps, transformer):
    json_details = {
        "model": model_name,
        "tokenizer": tokenizer_name,
        "timesteps": timesteps,
        "transformer": True
    }

    os.makedirs(output_directory_name, exist_ok=False)
    shutil.copyfile(model_name, os.path.join(output_directory_name, model_name))
    shutil.copyfile(tokenizer_name, os.path.join(output_directory_name, tokenizer_name))

    with open(os.path.join(output_directory_name, "details.json"), 'w') as f:
        json.dump(json_details, f, indent=4)

    shutil.copytree("midis_train", os.path.join(output_directory_name, "midis_train"))
    shutil.copytree("midis_test", os.path.join(output_directory_name, "midis_test"))
    shutil.copytree("midis_val", os.path.join(output_directory_name, "midis_val"))

    print(f"Model and tokenizer saved in {output_directory_name}")