import os
from pathlib import Path
import pickle
import sys
import cv2
import numpy as np

sys.path.append(str(Path(__file__).absolute().parents[5]))
from models.MoDE_Diffusion_Policy.calvin_env.calvin_env.utils.utils import to_relative_action


NUM_SEQS = 242


def get_all_shapes():
    print("Contents of calvin dataset:")
    calvin_path = f"{str(Path(__file__).absolute().parents[3])}/dataset/calvin_debug_dataset/training/episode_0358482.npz"
    with np.load(calvin_path) as data:
        for key in data.files:
            print(key, data[key].shape)

    print("\nContents of IRL kitchen dataset:")
    irl_kitchen_path = "/home/temp_store/troth/data/kit_irl_real_kitchen/lang/mdt_annotations/04_04_2024-15_53_21_0_17_79_banana_from_right_stove_to_sink_62/signal_dict.pickle"
    with open(irl_kitchen_path, "rb") as f:
        data = pickle.load(f)
        for key in data.keys():
            if key == "traj_length":
                continue
            if type(data[key][0]) == np.ndarray:
                print(key, data[key][0].shape)
            else:
                print(key, len(data[key]))


def get_imgs_from_detectron(split, seq_number):
    split_path = f"home/temp_store/troth/outputs/gripper_detection_split/eval/{split}"
    seq_name = os.listdir(split_path)[seq_number - 1]

    rgb_static = cv2.imread(f"{split_path}/{seq_name}/trajs/{seq_name}_cam_1_traj_00.png")
    rgb_gripper = cv2.imread(f"{split_path}/{seq_name}/trajs/{seq_name}_cam_2_traj_00.png")

    return rgb_static, rgb_gripper


def get_actions_and_obs_from_irl_kitchen(split, episode_number):
    split_path = f"/home/temp_store/troth/data/kit_irl_real_kitchen_split/lang/mdt_annotations/{split}"
    seq_name = os.listdir(split_path)[episode_number - 1]

    with open(f"{split_path}/{seq_name}/signal_dict.pickle", "rb") as f:
        data = pickle.load(f)

        actions = data["joint_state"] # TODO: not des_joint_state?
        robot_obs = np.ndarray()
        rel_actions = to_relative_action(actions, robot_obs)
        scene_obs = np.ndarray()
    
    return actions, rel_actions, robot_obs, scene_obs


def create_empty_rgb_and_depth_arrays():
    rgb_tactile = np.zeros((160, 120, 6))
    depth_static = np.zeros((200, 200))
    depth_gripper = np.zeros((84, 84))
    depth_tactile = np.zeros((160, 120, 2))

    return rgb_tactile, depth_static, depth_gripper, depth_tactile


def main():
    for split in ["training", "validation"]:
        output_path = f"{str(Path(__file__).absolute().parents[3])}/dataset/trajectory-planning/{split}"
        os.makedirs(output_path, exist_ok=True)

        for episode_number in range(NUM_SEQS):
            rgb_static, rgb_gripper = get_imgs_from_detectron(split, episode_number)
            actions, rel_actions, robot_obs, scene_obs = get_actions_and_obs_from_irl_kitchen(split, episode_number)
            rgb_tactile, depth_static, depth_gripper, depth_tactile = create_empty_rgb_and_depth_arrays()
            
            np.savez(f"{output_path}/episode_{episode_number:03d}.npz", actions=actions, rel_actions=rel_actions, robot_obs=robot_obs, scene_obs=scene_obs,
                    rgb_static=rgb_static, rgb_gripper=rgb_gripper, rgb_tactile=rgb_tactile,
                    depth_static=depth_static, depth_gripper=depth_gripper, depth_tactile=depth_tactile)


if __name__ == "__main__":
    #main()
    get_all_shapes()
