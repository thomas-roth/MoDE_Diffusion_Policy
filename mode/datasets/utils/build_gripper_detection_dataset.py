import os
import pickle
import sys
import cv2
import numpy as np

sys.path.append("/home/thomas/bt/bt-trajectory-planning")
from models.MoDE_Diffusion_Policy.calvin_env.calvin_env.utils.utils import to_relative_action


NUM_SEQS = 242


def get_all_shapes():
    print("Contents of calvin dataset:")
    calvin_path = "/home/thomas/bt/bt-trajectory-planning/models/MoDE_Diffusion_Policy/dataset/calvin_debug_dataset/training/episode_0358482.npz"
    with np.load(calvin_path) as data:
        for key in data.files:
            print(key, data[key].shape)

    print("\nContents of IRL kitchen dataset:")
    irl_kitchen_path = "/home/thomas/bt/bt-trajectory-planning/models/MoDE_Diffusion_Policy/dataset/gripper_detection/kit_irl_real_kitchen/lang/mdt_annotations/04_04_2024-15_53_21_0_17_79_banana_from_right_stove_to_sink_62/signal_dict.pickle"
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
    seqs_path = f"/media/SICHERUNGEN/bt/gripper_detection/eval/{split}"
    seq_name = os.listdir(seqs_path)[seq_number - 1]
    imgs_path = f"{seqs_path}/{seq_name}/trajs"

    rgb_static = cv2.imread(f"{imgs_path}/{seq_name}_cam_0_traj_00.png")
    rgb_gripper = cv2.imread(f"{imgs_path}/{seq_name}_cam_1_traj_00.png")

    return rgb_static, rgb_gripper


def get_actions_and_obs_from_irl_kitchen():
    # actions == (des_)joint_state (shape: (7,))
    # robot_obs == ?
    # rel_actions == to_relative_actions(actions, robot_obs)
    # scene_obs == ?

    actions = np.ndarray()
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
        output_path = f"/home/thomas/bt/bt-trajectory-planning/models/MoDE_Diffusion_Policy/dataset/gripper_detection/{split}"
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
