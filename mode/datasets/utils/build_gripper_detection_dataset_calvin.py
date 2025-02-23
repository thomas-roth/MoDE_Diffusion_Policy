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


# can be changed by user
OUTPUT_DIR = "/home/troth/bt/data/gripper_detection_calvin"
TRAJ_SIMPLIFICATION_RDP_EPSILON = 0.01 # unit of world coordinates (meters?) => 0.01 = 1 cm?, 0.01 ≈ 7 points per trajectory, TODO: figure out world coords unit
LANG_COND_COORDS_PRECISION = 4
TRAJ_DRAWING_THICKNESS = 2 # pixels
TRAJ_DRAWING_CIRCLE_RADIUS = 5 # pixels
GIF_FRAME_QUANTIZATION_METHOD = 2 # enum of size 4, 2 = Image.FASTOCTREE (fast but less accurate)
GIF_FRAME_QUANTIZATION_KMEANS = 1 # cluster changes of pixels allowed per kmeans iteration => lower = try harder to find best color palette, 0 = no clustering
GIF_DURATION = 67 # ms per frame => 67 ≈ 15 fps
GIF_NUM_LOOPS = 0 # 0 = infinite
DATASET_PATH = "/DATA/calvin/task_D_D"
AUTO_LANG_ANN_FOLDER = "lang_clip_resnet50"

# don't touch
CALVIN_GRIPPER_WIDTH_OPEN = 1.0
CALVIN_GRIPPER_WIDTH_CLOSED = -1.0


logging.basicConfig(filename=f"{OUTPUT_DIR}/build_dataset.log", level=logging.INFO, format="%(asctime)s - %(message)s", filemode='w')


def simplify_trajectory(gripper_centers, gripper_widths):
    # Ramer-Douglas-Peucker algorithm

    assert len(gripper_centers) == len(gripper_widths)

    simplification_mask = rdp(gripper_centers, epsilon=TRAJ_SIMPLIFICATION_RDP_EPSILON, return_mask=True)

    gripper_centers_simplified = gripper_centers[simplification_mask]
    gripper_widths_simplified = gripper_widths[simplification_mask]

    return gripper_centers_simplified, gripper_widths_simplified


def build_lang_conditioning(gripper_centers, gripper_widths):
    # gripper centers in world space => 3D

    is_gripper_open = gripper_widths[0] == CALVIN_GRIPPER_WIDTH_OPEN

    conditioning_contents = []
    for (gripper_center, gripper_width) in zip(gripper_centers, gripper_widths):
        # TODO?: normalize/scale (values between -1 and 1, but mostly around 0)
        rounded_gripper_center_x = round(gripper_center[0], LANG_COND_COORDS_PRECISION)
        rounded_gripper_center_y = round(gripper_center[1], LANG_COND_COORDS_PRECISION)
        rounded_gripper_center_z = round(gripper_center[2], LANG_COND_COORDS_PRECISION)

        conditioning_contents.append(f"({rounded_gripper_center_x}, {rounded_gripper_center_y}, {rounded_gripper_center_z})")
        
        if is_gripper_open and gripper_width == CALVIN_GRIPPER_WIDTH_CLOSED:
            conditioning_contents.append("<action>Close Gripper</action>")
            is_gripper_open = False
        elif not is_gripper_open and gripper_width == CALVIN_GRIPPER_WIDTH_OPEN:
            conditioning_contents.append("<action>Open Gripper</action>")
            is_gripper_open = True
    
    return "<ans>[" + str.join(", ", conditioning_contents) + "]</ans>"


def project_gripper_centers_to_cam(env, gripper_centers_world, cam_id):
    # TODO: handle rare massive outliers in gripper cam

    # fix different names of projection & view matrices between static & gripper cam
    env.cameras[1].projectionMatrix = env.cameras[1].projection_matrix
    del env.cameras[1].projection_matrix
    env.cameras[1].viewMatrix = env.cameras[1].view_matrix
    del env.cameras[1].view_matrix

    gripper_centers_world_ones = np.c_[np.array(gripper_centers_world), np.ones(len(gripper_centers_world))]
    gripper_centers_projected = env.cameras[cam_id].project(gripper_centers_world_ones.T)

    return np.transpose(gripper_centers_projected)


def draw_trajectory(img, gripper_centers, gripper_widths):
    # gripper centers in image space => 2D

    img_copy = img.copy()

    is_gripper_open = gripper_widths[0] == CALVIN_GRIPPER_WIDTH_OPEN

    for i in range(len(gripper_centers) - 1):
        color = (round((i+1) / len(gripper_centers) * 255), 0, 0) # black to red over time
        cv2.line(img_copy, tuple(gripper_centers[i]), tuple(gripper_centers[i+1]), color, thickness=TRAJ_DRAWING_THICKNESS)

        if is_gripper_open and gripper_widths[i] == CALVIN_GRIPPER_WIDTH_CLOSED:
            # close gripper => green circle
            cv2.circle(img_copy, tuple(gripper_centers[i]), radius=TRAJ_DRAWING_CIRCLE_RADIUS, color=(0, 255, 0), thickness=TRAJ_DRAWING_THICKNESS)
            is_gripper_open = False
        elif not is_gripper_open and gripper_widths[i] == CALVIN_GRIPPER_WIDTH_OPEN:
            # open gripper => blue circle
            cv2.circle(img_copy, tuple(gripper_centers[i]), radius=TRAJ_DRAWING_CIRCLE_RADIUS, color=(0, 0, 255), thickness=TRAJ_DRAWING_THICKNESS)
            is_gripper_open = True

    return img_copy


def save_imgs_gifs_to_disk(imgs_all_seqs, dataset_split, save_first_img_per_seq, save_gif_per_seq):
    seq_loop_tqdm_desc_contents = []
    if save_first_img_per_seq:
        imgs_dir = f"{OUTPUT_DIR}/imgs/{dataset_split}"
        os.makedirs(imgs_dir, exist_ok=True)
        seq_loop_tqdm_desc_contents.append("images")
    if save_gif_per_seq:
        gifs_dir = f"{OUTPUT_DIR}/gifs/{dataset_split}"
        os.makedirs(gifs_dir, exist_ok=True)
        seq_loop_tqdm_desc_contents.append("gifs")
    seq_loop_tqdm_desc = "Saving " + " and ".join(seq_loop_tqdm_desc_contents) + " to disk"
    
    num_digits = len(str(len(imgs_all_seqs)))

    for i, imgs_anno_per_seq in tqdm(enumerate(imgs_all_seqs), total=len(imgs_all_seqs), desc=seq_loop_tqdm_desc):
        imgs_seq = imgs_anno_per_seq["imgs"]
        anno_seq = imgs_anno_per_seq["anno"]

        for cam_name in ["rgb_static", "rgb_gripper"]:
            file_name = f"{i:0{num_digits}d}_{anno_seq}_{cam_name.split('_')[1]}"

            if save_first_img_per_seq:
                first_img = Image.fromarray(imgs_seq[cam_name][0])

                first_img.save(f"{imgs_dir}/{file_name}.png")

            if save_gif_per_seq:
                gif_frames = []
                for img in imgs_seq[cam_name]:
                    quantized_img = Image.fromarray(img).quantize(method=GIF_FRAME_QUANTIZATION_METHOD, kmeans=GIF_FRAME_QUANTIZATION_KMEANS) # to reduce file size
                    gif_frames.append(quantized_img)

                gif_frames[0].save(f"{gifs_dir}/{file_name}.gif", save_all=True, append_images=gif_frames[1:],
                                   duration=GIF_DURATION, loop=GIF_NUM_LOOPS)


def build_conds_and_imgs(dataset_split, save_first_img_per_seq, save_gif_per_seq):
    env_conf = OmegaConf.load(f"{DATASET_PATH}/{dataset_split}/.hydra/merged_config.yaml")
    del env_conf.cameras["tactile"] # not relevant for this task & breaks hydra instantiation
    env = hydra.utils.instantiate(env_conf.env, use_vr=False, use_scene_info=True)

    calvin_root = Path(__file__).absolute().parents[3] / "calvin_env"
    dataloader = CalvinDataLoader(calvin_root, f"{DATASET_PATH}/{dataset_split}")
    
    lengths_simplified_trajs = []

    lang_conds_all_seqs = []
    imgs_all_seqs = []
    for seq in tqdm(dataloader, total=len(dataloader), desc=f"Building dataset for {dataset_split} split"):
        assert len(seq["obs"]["robot_obs"]) == len(seq["obs"]["rel_actions"]) == len(seq["obs"]["rgb_static"]) == len(seq["obs"]["rgb_gripper"])

        gripper_centers_world = np.array(seq["obs"]["robot_obs"])[:, :3]
        gripper_widths = np.array(seq["obs"]["robot_obs"])[:, -1]

        # reset to start of sequence
        env.reset(robot_obs=seq["obs"]["robot_obs"][0], scene_obs=seq["obs"]["scene_obs"][0])

        # simplify trajectory of center points in world space
        simplified_gripper_centers_world, simplified_gripper_widths = simplify_trajectory(gripper_centers_world, gripper_widths)
        lengths_simplified_trajs.append(len(simplified_gripper_centers_world))

        # build lang conditioning for simplified trajectory in world space
        lang_cond_per_seq = build_lang_conditioning(simplified_gripper_centers_world, simplified_gripper_widths)
        lang_conds_all_seqs.append(lang_cond_per_seq)

        # project simplified trajectory to image spaces & draw on images
        imgs_per_seq = {"rgb_static": [], "rgb_gripper": []}
        for cam_id, cam_name in enumerate(["rgb_static", "rgb_gripper"]):            
            for timestep in range(len(seq["obs"]["robot_obs"])):
                # project gripper centers to both cams for first timestep
                # only update gripper cam for each timestep (only gripper cam moves)
                if timestep == 0 or cam_name == "rgb_gripper":
                    env.step(seq["obs"]["rel_actions"][timestep]) # move gripper to position at timestep to update camera view matrix
                    simplified_gripper_centers_projected = project_gripper_centers_to_cam(env, simplified_gripper_centers_world, cam_id)

                img = seq["obs"][cam_name][timestep]
                img_with_traj = draw_trajectory(img, simplified_gripper_centers_projected, simplified_gripper_widths)

                imgs_per_seq[cam_name].append(img_with_traj)

                if not save_gif_per_seq:
                    # only first frame of each sequence needed
                    break
        imgs_all_seqs.append({"imgs": imgs_per_seq, "anno": seq["anno"]})

    logging.info(f"Average trajectory length: {np.mean(lengths_simplified_trajs)}")

    if save_first_img_per_seq or save_gif_per_seq:
        save_imgs_gifs_to_disk(imgs_all_seqs, dataset_split, save_first_img_per_seq, save_gif_per_seq)

    return lang_conds_all_seqs, imgs_all_seqs


def build_new_auto_lang_ann(lang_conds, dataset_split, timestamp):
    lang_cond_explanation = "Use the following list of tuples enclosed by <ans> and </ans> tags as a guide for the trajectory of the end effector. " \
                            "A tuple (x, y, z) denotes the 3D location of the end effector in world space. The tags <action> and </action> enclose a gripper action"

    auto_lang_ann = np.load(f"{DATASET_PATH}/{dataset_split}/{AUTO_LANG_ANN_FOLDER}/auto_lang_ann.npy", allow_pickle=True)
    for i, (task, lang_cond) in enumerate(zip(auto_lang_ann[np.newaxis][0]["language"]["ann"], lang_conds)):
        auto_lang_ann[np.newaxis][0]["language"]["ann"][i] = f"{task}. {lang_cond_explanation}: {lang_cond}"

    lang_annos_output_dir = f"{OUTPUT_DIR}/{AUTO_LANG_ANN_FOLDER}/{timestamp}/{dataset_split}"
    os.makedirs(lang_annos_output_dir, exist_ok=True)

    np.save(f"{lang_annos_output_dir}/auto_lang_ann.npy", auto_lang_ann)

    # TODO: update embeddings (inference with lang clip?)

    print(f"Built new auto lang annotations for {dataset_split} split")


def build_dataset(save_first_img_per_seq=False, save_gif_per_seq=False):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for dataset_split in ["training", "validation"]:
        lang_conds, _ = build_conds_and_imgs(dataset_split, save_first_img_per_seq, save_gif_per_seq)
        build_new_auto_lang_ann(lang_conds, dataset_split, timestamp)


if __name__ == '__main__':
    build_dataset(save_first_img_per_seq=True, save_gif_per_seq=True)
