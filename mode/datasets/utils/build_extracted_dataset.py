import os
import shutil
import numpy as np
from tqdm import tqdm


dataset_path = "/home/thomas/bt/bt-trajectory-planning/models/MoDE_Diffusion_Policy/dataset/calvin_debug_dataset"
dataset_path = "/media/thomas/SICHERUNGEN/hiwi/task_D_D"

for split in ["training", "validation"]:
    print(f"\nProcessing split '{split}'")

    split_path = os.path.join(dataset_path, split)

    output_path = os.path.join(split_path, "extracted")
    lang_annos_path = os.path.join(split_path, "lang_annotations")
    os.makedirs(output_path, exist_ok=True)

    # Get episode numbers & copy .npz files to extracted folder
    episode_numbers = []
    for file in tqdm(os.listdir(split_path), total=len(os.listdir(split_path)), desc=f"Copying .npz files"):
        if file.endswith(".npz"):
            episode_number = file.split(".")[0].split("_")[1]
            episode_numbers.append(episode_number)

            # copy .npz files to extracted folder
            shutil.copy(os.path.join(split_path, file), os.path.join(output_path, file))
    
    # Save episode numbers to ep_npz_names.list
    episode_numbers.sort()
    list_file_path = os.path.join(output_path, "ep_npz_names.list")
    with open(list_file_path, "w") as f:
        f.write("\n".join([str(ep) for ep in episode_numbers]))
    
    # Generate ep_rel_actions.npy
    rel_actions = []
    for episode_number in tqdm(episode_numbers, total=len(episode_numbers), desc="Generating ep_rel_actions.npy"):
        episode_file = os.path.join(output_path, f"episode_{episode_number}.npz")
        with np.load(episode_file) as data:
            if "rel_actions" in data:
                rel_actions.append(data["rel_actions"])
            else:
                print(f"Warning: Episode {episode_number} has no rel actions")
    
    if rel_actions:
        rel_actions_array = np.concatenate(rel_actions, axis=0)
        np.save(os.path.join(output_path, "ep_rel_actions.npy"), rel_actions_array)
    else:
        print("Warning: No rel actions found in any episode")
