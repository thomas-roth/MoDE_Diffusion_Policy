import os
import sys
import logging
import numpy as np
import cv2
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from rdp import rdp

sys.path.append(str(Path(__file__).absolute().parents[5]))
from models.MoDE_Diffusion_Policy.mode.datasets.utils.calvin_dataloader import CalvinDataLoader



DATASET_PATH = "/DATA/calvin/task_D_D"
OUTPUT_DIR = "/home/troth/bt/data/gripper_detection_calvin"


logging.basicConfig(filename=f"{OUTPUT_DIR}/build_dataset.log", level=logging.INFO, format="%(asctime)s - %(message)s", filemode='w')


def get_tcp_center_screen(obs_batch, cam):
    tcp_pos_world = np.array(obs_batch["robot_obs"])[:, :3]
    tcp_pos_world_ones = np.c_[np.array(tcp_pos_world), np.ones(len(tcp_pos_world))]

    tcp_center_gripper_screen = cam.project(tcp_pos_world_ones.T)
    
    tcp_center_gripper_screen = [np.array([x, y]) for x, y in zip(tcp_center_gripper_screen[0], tcp_center_gripper_screen[1])]
    tcp_center_gripper_screen = np.stack(tcp_center_gripper_screen)

    return tcp_center_gripper_screen


def simplify_trajectory(tcp_centers, gripper_widths, epsilon=1):
    # Ramer-Douglas-Peucker algorithm

    mask = rdp(tcp_centers, epsilon=epsilon, return_mask=True)

    tcp_centers_simplified = tcp_centers[mask]
    gripper_widths_simplified = gripper_widths[mask]
    
    assert len(tcp_centers_simplified) == len(gripper_widths_simplified)

    return tcp_centers_simplified, gripper_widths_simplified


def build_lang_conditioning(tcp_centers, gripper_widths, img_size):
    gripper_width_open = 1.0
    gripper_width_closed = -1.0

    is_gripper_open = gripper_widths[0] == gripper_width_open

    conditioning_contents = []

    for (tcp_center, gripper_width) in zip(tcp_centers, gripper_widths):
        # normalize coordinates to [0, 1] to enable different image sizes
        tcp_center_normalized = (float(tcp_center[0]) / img_size, float(tcp_center[1]) / img_size)

        conditioning_contents.append(f"({tcp_center_normalized[0]}, {tcp_center_normalized[1]})")

        if is_gripper_open and gripper_width == gripper_width_closed:
            conditioning_contents.append("<action>Close Gripper</action>")
        elif not is_gripper_open and gripper_width == gripper_width_open:
            conditioning_contents.append("<action>Open Gripper</action>")
    
    return "<ans>[" + str.join(", ", conditioning_contents) + "]</ans>"


def draw_trajectory(tcp_centers, gripper_widths, img):
    img_copy = img.copy()

    gripper_width_open = 1.0
    gripper_width_closed = -1.0

    is_gripper_open = gripper_widths[0] == gripper_width_open

    for i in range(len(tcp_centers) - 1):
        color = (round((i+1) / len(tcp_centers) * 255), 0, 0) # black to red over time
        cv2.line(img_copy, tuple(tcp_centers[i]), tuple(tcp_centers[i+1]), color, thickness=2)

        if is_gripper_open and gripper_widths[i] == gripper_width_closed:
            # close gripper => green circle
            cv2.circle(img_copy, tuple(tcp_centers[i]), radius=5, color=(0, 255, 0), thickness=2)
            is_gripper_open = False
        elif not is_gripper_open and gripper_widths[i] == gripper_width_open:
            # open gripper => blue circle
            cv2.circle(img_copy, tuple(tcp_centers[i]), radius=5, color=(0, 0, 255), thickness=2)
            is_gripper_open = True

    return img_copy


def save_imgs_to_disk(data, save_gifs):
    num_digits = len(str(len(data) // 2))

    os.makedirs(f"{OUTPUT_DIR}/trajs", exist_ok=True)
    if save_gifs:
        os.makedirs(f"{OUTPUT_DIR}/gifs", exist_ok=True)

    for i, (imgs_seq, anno, cam_id) in tqdm(enumerate(data), total=len(data), desc="Saving images to disk"):
        cam_name = ['static', 'gripper'][cam_id]
        img_name = f"{i:0{num_digits}d}_{anno}_{cam_id}-{cam_name}"
        Image.fromarray(imgs_seq[0]).save(f"{OUTPUT_DIR}/trajs/{img_name}.jpg")

        if save_gifs:
            gif_frames = []
            for img in imgs_seq:
                gif_frames.append(Image.fromarray(img).quantize(colors=256, method=2, kmeans=1))

            gif_frames[0].save(f"{OUTPUT_DIR}/gifs/{img_name}.gif", save_all=True, append_images=gif_frames[1:], duration=75, loop=0)


def build_conds_and_imgs(split, save_imgs, save_gifs):
    env_conf = OmegaConf.load(f"{DATASET_PATH}/{split}/.hydra/merged_config.yaml")
    del env_conf.cameras["tactile"] # not relevant for this task & breaks hydra instantiation
    env = hydra.utils.instantiate(env_conf.env, use_vr=False, use_scene_info=True)

    calvin_root = Path(__file__).absolute().parents[3] / "calvin_env"
    seq_len = 1024
    step_size = 2
    dataloader = CalvinDataLoader(calvin_root, f"{DATASET_PATH}/{split}", seq_len=seq_len, stepsize=step_size)
    
    simplified_traj_lengths = []

    lang_conds = []
    imgs_data = []

    num_of_seqs = len(dataloader.annotations["info"]["indx"])
    for i in tqdm(range(num_of_seqs), total=num_of_seqs, desc=f"Building dataset for {split} split"):
        obs_task, anno, _ = dataloader.get_single_problem(problem_index=i)
        gripper_widths = np.array(obs_task["robot_obs"])[:, -1]

        # reset to start of sequence
        env.reset(robot_obs=obs_task["robot_obs"][0], scene_obs=obs_task["scene_obs"][0])
        
        # Fix different names of projection & view matrices between static & gripper cam
        env.cameras[1].projectionMatrix = env.cameras[1].projection_matrix
        del env.cameras[1].projection_matrix
        env.cameras[1].viewMatrix = env.cameras[1].view_matrix
        del env.cameras[1].view_matrix

        for cam_id in [0, 1]:
            tcp_centers = get_tcp_center_screen(obs_task, cam=env.cameras[cam_id])

            tcp_centers_simplified, gripper_widths_simplified = simplify_trajectory(tcp_centers, gripper_widths)
            simplified_traj_lengths.append(len(tcp_centers_simplified))

            assert env.cameras[cam_id].width == env.cameras[cam_id].height
            lang_cond = build_lang_conditioning(tcp_centers_simplified, gripper_widths_simplified, env.cameras[cam_id].width)
            lang_conds.append(lang_cond)

            datapoint = ([], anno, cam_id)
            for img in [obs_task["rgb_static"], obs_task["rgb_gripper"]][cam_id]:
                # FIXME: trajs for gripper cam not visible bc cam moves => new projection per frame?

                img_with_traj = draw_trajectory(tcp_centers_simplified, gripper_widths_simplified, img)
                datapoint[0].append(img_with_traj)

                if not save_gifs:
                    # only first frame of each sequence needed
                    break
            
            imgs_data.append(datapoint)

    logging.info(f"Average trajectory length: {np.mean(simplified_traj_lengths)}")

    if save_imgs:
        save_imgs_to_disk(imgs_data, save_gifs)

    return lang_conds, imgs_data


def build_auto_lang_ann_with_lang_conditionings(split, lang_conds):
    lang_cond_explanation = "Use the following list of tuples enclosed by <ans> and </ans> tags as a guide for the trajectory of the end effector. " \
                            "The tuple denotes the relative x and y location of the end effector in the image. The action tags indicate the gripper action"

    auto_lang_ann = np.load(f"{DATASET_PATH}/{split}/lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    for i, (task, lang_cond) in enumerate(zip(auto_lang_ann[np.newaxis][0]["language"]["ann"], lang_conds)):
        auto_lang_ann[np.newaxis][0]["language"]["ann"][i] = f"{task}. {lang_cond_explanation}: {lang_cond}"

    np.save(f"{OUTPUT_DIR}/auto_lang_ann_{split}.npy", auto_lang_ann)


def build_dataset(save_imgs=False, save_gifs=False):
    for split in ["training", "validation"]:
        lang_conds, _ = build_conds_and_imgs(split, save_imgs, save_gifs)
        build_auto_lang_ann_with_lang_conditionings(split, lang_conds)


if __name__ == '__main__':
    build_dataset(save_imgs=False, save_gifs=False)
