import datetime
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
LANG_COND_COORD_PRECISION = 4


logging.basicConfig(filename=f"{OUTPUT_DIR}/build_dataset.log", level=logging.INFO, format="%(asctime)s - %(message)s", filemode='w')


def project_gripper_centers_to_cam(env, gripper_centers_world, cam_id):
    gripper_centers_world_ones = np.c_[np.array(gripper_centers_world), np.ones(len(gripper_centers_world))]

    projected_gripper_centers = env.cameras[cam_id].project(gripper_centers_world_ones.T)
    projected_gripper_centers = [np.array([x, y]) for x, y in zip(projected_gripper_centers[0], projected_gripper_centers[1])]
    projected_gripper_centers = np.stack(projected_gripper_centers)

    return projected_gripper_centers


def simplify_trajectory(gripper_centers, gripper_widths, epsilon=0.01):
    # Ramer-Douglas-Peucker algorithm

    mask = rdp(gripper_centers, epsilon=epsilon, return_mask=True)

    gripper_centers_simplified = gripper_centers[mask]
    gripper_widths_simplified = gripper_widths[mask]
    
    assert len(gripper_centers_simplified) == len(gripper_widths_simplified)

    return gripper_centers_simplified, gripper_widths_simplified


def build_lang_conditioning(gripper_centers, gripper_widths):
    # gripper centers in world space => 3D

    gripper_width_open = 1.0
    gripper_width_closed = -1.0

    is_gripper_open = gripper_widths[0] == gripper_width_open

    conditioning_contents = []

    for (gripper_center, gripper_width) in zip(gripper_centers, gripper_widths):
        # TODO?: normalize? (values between -2 and 4 (really? mostly saw around 0))
        rounded_gripper_center_x = round(gripper_center[0], LANG_COND_COORD_PRECISION)
        rounded_gripper_center_y = round(gripper_center[1], LANG_COND_COORD_PRECISION)
        rounded_gripper_center_z = round(gripper_center[2], LANG_COND_COORD_PRECISION)

        conditioning_contents.append(f"({rounded_gripper_center_x}, {rounded_gripper_center_y}, {rounded_gripper_center_z})")
        
        if is_gripper_open and gripper_width == gripper_width_closed:
            conditioning_contents.append("<action>Close Gripper</action>")
            is_gripper_open = False
        elif not is_gripper_open and gripper_width == gripper_width_open:
            conditioning_contents.append("<action>Open Gripper</action>")
            is_gripper_open = True
    
    return "<ans>[" + str.join(", ", conditioning_contents) + "]</ans>"


def draw_trajectory(img, gripper_centers, gripper_widths):
    # gripper centers in image space => 2D

    img_copy = img.copy()

    # defined in calvin env
    gripper_width_open = 1.0
    gripper_width_closed = -1.0

    is_gripper_open = gripper_widths[0] == gripper_width_open

    for i in range(len(gripper_centers) - 1):
        color = (round((i+1) / len(gripper_centers) * 255), 0, 0) # black to red over time
        cv2.line(img_copy, tuple(gripper_centers[i]), tuple(gripper_centers[i+1]), color, thickness=2)

        if is_gripper_open and gripper_widths[i] == gripper_width_closed:
            # close gripper => green circle
            cv2.circle(img_copy, tuple(gripper_centers[i]), radius=5, color=(0, 255, 0), thickness=2)
            is_gripper_open = False
        elif not is_gripper_open and gripper_widths[i] == gripper_width_open:
            # open gripper => blue circle
            cv2.circle(img_copy, tuple(gripper_centers[i]), radius=5, color=(0, 0, 255), thickness=2)
            is_gripper_open = True

    return img_copy


def save_imgs_to_disk(imgs_all_seqs, split, save_gifs):
    num_digits = len(str(len(imgs_all_seqs)))

    os.makedirs(f"{OUTPUT_DIR}/trajs/{split}", exist_ok=True)
    if save_gifs:
        os.makedirs(f"{OUTPUT_DIR}/gifs/{split}", exist_ok=True)        

    for i, imgs_anno_per_seq in tqdm(enumerate(imgs_all_seqs), total=len(imgs_all_seqs), desc="Saving images to disk"):
        imgs_seq = imgs_anno_per_seq["imgs"]
        anno_seq = imgs_anno_per_seq["anno"]

        for cam_name in ["rgb_static", "rgb_gripper"]:
            img_name = f"{i:0{num_digits}d}_{anno_seq}_{cam_name}"

            # save first frame of each sequence
            Image.fromarray(imgs_seq[cam_name][0]).save(f"{OUTPUT_DIR}/trajs/{split}/{img_name}.jpg")

            if save_gifs:
                # save all frames of each sequence as gifs
                gif_frames = []
                for img in imgs_seq[cam_name]:
                    gif_frames.append(Image.fromarray(img).quantize(colors=256, method=2, kmeans=1))

                gif_frames[0].save(f"{OUTPUT_DIR}/gifs/{split}/{img_name}.gif", save_all=True, append_images=gif_frames[1:], duration=75, loop=0)


def position_cameras_to_timestep(env, rel_action_timestep):
    # move gripper to position at timestep to update camera view matrix
    env.step(rel_action_timestep)

    # fix different names of projection & view matrices between static & gripper cam
    env.cameras[1].projectionMatrix = env.cameras[1].projection_matrix
    del env.cameras[1].projection_matrix
    env.cameras[1].viewMatrix = env.cameras[1].view_matrix
    del env.cameras[1].view_matrix


def build_conds_and_imgs(split, save_imgs, save_gifs):
    env_conf = OmegaConf.load(f"{DATASET_PATH}/{split}/.hydra/merged_config.yaml")
    del env_conf.cameras["tactile"] # not relevant for this task & breaks hydra instantiation
    env = hydra.utils.instantiate(env_conf.env, use_vr=False, use_scene_info=True)

    calvin_root = Path(__file__).absolute().parents[3] / "calvin_env"
    seq_len = 1024
    step_size = 2
    dataloader = CalvinDataLoader(calvin_root, f"{DATASET_PATH}/{split}", seq_len=seq_len, stepsize=step_size)
    
    lengths_simplified_trajs = []

    lang_conds_all_seqs = []
    imgs_all_seqs = []

    num_of_seqs = len(dataloader.annotations["info"]["indx"]) # FIXME: num_of_seqs different per split
    for i in tqdm(range(num_of_seqs), total=num_of_seqs, desc=f"Building dataset for {split} split"):
        if i == 20:
            break

        obs_seq, anno_seq, _ = dataloader.get_single_problem(problem_index=i)
        assert len(obs_seq["robot_obs"]) == len(obs_seq["rel_actions"]) == len(obs_seq["rgb_static"]) == len(obs_seq["rgb_gripper"])

        gripper_centers_world = np.array(obs_seq["robot_obs"])[:, :3]
        gripper_widths = np.array(obs_seq["robot_obs"])[:, -1]

        # reset to start of sequence
        env.reset(robot_obs=obs_seq["robot_obs"][0], scene_obs=obs_seq["scene_obs"][0])

        # simplify trajectory of center points in world space
        simplified_gripper_centers_world, simplified_gripper_widths = simplify_trajectory(gripper_centers_world, gripper_widths)
        lengths_simplified_trajs.append(len(simplified_gripper_centers_world))

        # build lang conditioning for simplified trajectory in world space
        lang_cond_per_seq = build_lang_conditioning(simplified_gripper_centers_world, simplified_gripper_widths)
        lang_conds_all_seqs.append(lang_cond_per_seq)

        # project simplified trajectory to image spaces & draw on images
        imgs_per_seq = {"rgb_static": [], "rgb_gripper": []}
        for cam_id, cam_name in enumerate(["rgb_static", "rgb_gripper"]):            
            for timestep in range(len(obs_seq["robot_obs"])):
                if timestep == 0 or cam_name == "rgb_gripper":
                    # only update gripper cam for each timestep as only it moves
                    position_cameras_to_timestep(env, obs_seq["rel_actions"][timestep])
                    simplified_gripper_centers_projected = project_gripper_centers_to_cam(env, simplified_gripper_centers_world, cam_id)

                img = obs_seq[cam_name][timestep]
                img_with_traj = draw_trajectory(img, simplified_gripper_centers_projected, simplified_gripper_widths)

                imgs_per_seq[cam_name].append(img_with_traj)

            if not save_gifs:
                # only first frame of each sequence needed
                break
        imgs_all_seqs.append({"imgs": imgs_per_seq, "anno": anno_seq})

    logging.info(f"Average trajectory length: {np.mean(lengths_simplified_trajs)}")

    if save_imgs:
        save_imgs_to_disk(imgs_all_seqs, split, save_gifs)

    return lang_conds_all_seqs, imgs_all_seqs


def build_auto_lang_ann_with_lang_conditionings(split, lang_conds, timestamp):
    lang_cond_explanation = "Use the following list of tuples enclosed by <ans> and </ans> tags as a guide for the trajectory of the end effector. " \
                            "A tuple (x, y, z) denotes the 3D location of the end effector in world space. The tags <action> and </action> enclose a gripper action"

    auto_lang_ann = np.load(f"{DATASET_PATH}/{split}/lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    for i, (task, lang_cond) in enumerate(zip(auto_lang_ann[np.newaxis][0]["language"]["ann"], lang_conds)):
        auto_lang_ann[np.newaxis][0]["language"]["ann"][i] = f"{task}. {lang_cond_explanation}: {lang_cond}"

    lang_annotations_output_dir = f"{OUTPUT_DIR}/lang_annotations/{timestamp}/{split}"
    os.makedirs(lang_annotations_output_dir, exist_ok=True)

    np.save(f"{lang_annotations_output_dir}/auto_lang_ann.npy", auto_lang_ann)

    # TODO: update embeddings (inference with lang clip?)


def build_dataset(save_imgs=False, save_gifs=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for split in ["training", "validation"]:
        lang_conds, _ = build_conds_and_imgs(split, save_imgs, save_gifs)
        build_auto_lang_ann_with_lang_conditionings(split, lang_conds, timestamp)


if __name__ == '__main__':
    build_dataset(save_imgs=True, save_gifs=True)
