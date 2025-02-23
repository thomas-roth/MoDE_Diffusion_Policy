import os
import numpy as np
from torch.utils.data import Dataset


# TODO: remove unused parts after generic & specific dataset builders finished
class CalvinDataLoader(Dataset):
    def __init__(self, calvin_root, dataset_path, seq_len=128, step_size=2, annotations_file="lang_annotations/auto_lang_ann.npy"):
        self.calvin_root = calvin_root
        self.seq_len = seq_len
        self.dataset_path = os.path.join(self.calvin_root, dataset_path)
        self.step_size = step_size

        self.key_state_indices = None
        self.key_state_indices_global = None

        self.iterate_over_single_seqs = True
        self.continuous_reverse_stream = True

        self.annotations = np.load(os.path.join(self.dataset_path, annotations_file), allow_pickle=True).item()

        try:
            self.indices = next(iter(np.load(f"{dataset_path}/scene_info.npy", allow_pickle=True).item().values()))
            self.indices = list(range(53819, self.indices[-1] + 1))
            self.indices = self.indices[::self.step_size]
        except:
            self.indices = None

        self.key_state_start_indices = np.array([index[0] for index in self.annotations['info']['indx']])
        self.key_state_stop_indices = np.array([index[1] for index in self.annotations['info']['indx']])

        if self.indices is not None:
            self.current_goal_index = len(self.indices) - 1

        if "debug" in dataset_path:
            if self.indices is not None:
                self.current_goal_index = np.abs((np.array(self.indices) - self.key_state_start_indices[1])).argmin()
            self.seq_len = 128
        
        if self.indices is not None:
            self.last_indices = self.indices
        
        self.nl_annotations = self.annotations['language']['ann']
        self.task_annotations = list(zip(self.annotations['info']['indx'], self.annotations['language']['task']))

        self.task_nlp_map = dict()
        self.build_task_nlp_map()

        self.ds_indices = np.array([np.arange(start_index, start_index + (self.seq_len * self.step_size)) for start_index in self.key_state_start_indices])


    def __len__(self):
        return len(self.task_annotations)
    

    def __getitem__(self, index):
        if self.iterate_over_single_seqs:
            data = self.get_single_seq(index)
        elif self.continuous_reverse_stream:
            data = self.load_next_seqs_batch()
        else:
            data = self.load_seqs_batch(index)

        return {**data} # shallow copy


    def build_task_nlp_map(self):
        for task, ann in zip(self.annotations['language']['task'], self.nl_annotations):
            curr_task = task
            self.task_nlp_map.setdefault(curr_task, []).append(ann)
        
        self.task_nlp_map = {key: list(np.unique(value)) for key, value in self.task_nlp_map.items()}
    

    def get_possible_actions(self):
        return self.task_nlp_map
    

    def get_possible_annotations(self):
        return self.annotations['language']['ann']


    def build_dataset(self):
        """
        Generates a list of indices with each key state starting point as starting point. The number of samples for the
        horizon task is determined by self.seq_len and self.stepsize.
        """

        are_ds_indices_in_stop_indices = np.isin(self.ds_indices, self.key_state_stop_indices)
        self.key_state_indices = ([np.nonzero(boolean)[0] for boolean in are_ds_indices_in_stop_indices])
        self.key_state_indices_global = ([self.ds_indices[i, boolean] for i, boolean in enumerate(are_ds_indices_in_stop_indices)])

        nl_index = [np.where(np.isin(self.key_state_stop_indices, self.ds_indices[i, index]))[0] for i, index in enumerate(self.key_state_indices)]
        self.nl_annotations = [np.array(self.annotations['language']['ann'])[index] for index in nl_index]

        return self.ds_indices
    

    def set_last_goal_index(self, index):
        self.current_goal_index -= self.seq_len - index

    
    def get_global_goal_indices(self, indices):
        return np.array([self.indices[index] for index in indices])

    
    def is_finished(self):
        return self.current_goal_index <= self.indices[0]
    

    def load_next_seqs_batch(self):
        ds_path = self.dataset_path
        subset = "train"
        frames_static = []
        frames_gripper = []
        previous_task = ""
        key_state_indices = []
        task_annotations = []

        orig_data = []
        for i, index in enumerate(self.indices[self.current_goal_index - self.seq_len: self.current_goal_index]):
            if not os.path.exists(os.path.join(self.dataset_path, f"episode_{index:07d}.npz")):
                ds_path = os.path.join(os.path.dirname(ds_path), "validation")
                subset = "val"
            
            try:
                frame = np.load(os.path.join(ds_path, f"episode_{index:07d}.npz"), allow_pickle=True)
                orig_data.append(dict(frame))
                frames_static.append(frame["rgb_static"])
                frames_gripper.append(frame["rgb_gripper"])

                annotation = "Undefined"
                for (index_low, index_high), ann in self.task_annotations:
                    if index_low <= index <= index_high:
                        annotation = ann
                        if ann != previous_task:
                            key_state_indices.append((index_high - index_low) // self.step_size + i)
                            previous_task = ann
                
                task_annotations.append(annotation)
            except FileNotFoundError as e:
                subset = "train"
                print(e)
                pass
        
        orig_data = {key: [dic[key] for dic in orig_data] for key in orig_data[0]}
        orig_data["task_annotations"] = np.array(task_annotations)
        orig_data["key_state_indices"] = key_state_indices
        orig_data["key_state_labels"] = []
        orig_data["subset"] = subset

        return orig_data


    def load_seqs_batch(self, batch_index):
        frames_static = []
        task_annotations = []
        key_state_indices = []
        key_state_labels = []
        annotations = []

        for i, index in enumerate(self.ds_indices[batch_index]):
            if i % self.step_size != 0 and index not in self.key_state_stop_indices:
                continue

            try:
                frame = np.load(os.path.join(self.dataset_path, f"episode_{index:07d}.npz"), allow_pickle=True)
                frames_static.append(frame["rgb_static"])

                if index in self.key_state_stop_indices:
                    task_index = np.where(index == self.key_state_stop_indices)[0]
                    
                    task_annotations.append(self.annotations["language"]["task"][task_index.item()])
                    key_state_indices.append(len(frames_static) - 1)
                    key_state_labels.append(task_index.item())
                    annotations.append(self.nl_annotations[task_index.item()])
            except FileNotFoundError as e:
                print(e)
                pass
        
        annotations = np.concatenate([np.array(annotations[i]).repeat(index - (key_state_indices[i - 1] if i > 0 else 0)) for i, index in enumerate(key_state_indices)])
        task_annotations = np.concatenate([np.array(task_annotations[i]).repeat(index - (key_state_indices[i - 1] if i > 0 else 0)) for i, index in enumerate(key_state_indices)])

        # Fill last elements with (uncompleted) tasks
        undefined_annotation = self.nl_annotations[np.argmin(np.abs(self.key_state_start_indices - index))]
        undefined_task_anotation = self.annotations["language"]["task"][np.argmin(np.abs(self.key_state_start_indices - index))]
        undefined_annotations = np.full(len(frames_static) - annotations.shape[0], undefined_annotation)
        undefined_task_annotations = np.full(len(frames_static) - task_annotations.shape[0], undefined_task_anotation)
        annotations = np.concatenate([annotations, undefined_annotations])
        task_annotations = np.concatenate([task_annotations, undefined_task_annotations])

        assert annotations.shape[0] == len(frames_static)

        return {"frames": frames_static, "annotations": annotations, "task_annotations": task_annotations, "key_state_indices": key_state_indices, "key_state_labels": key_state_labels}


    def get_single_seq(self, seq_index):
        anno_seq_start_index, anno_seq_stop_index = self.annotations["info"]["indx"][seq_index]
        anno_seq = self.annotations["language"]["task"][seq_index]

        obs_seq = []
        for index in range(anno_seq_start_index, anno_seq_stop_index):
            try:
                frame = np.load(os.path.join(self.dataset_path, f"episode_{index:07d}.npz"), allow_pickle=True)
                obs_seq.append(dict(frame))
            except FileNotFoundError as e:
                print(e)
                pass
        
        obs_seq = {key: [dic[key] for dic in obs_seq] for key in obs_seq[0]}

        return {"obs": obs_seq, "anno": anno_seq}


    def get_n_seqs(self, start_seq_index, n_seqs):
        start_indices = np.array([index[0] for index in self.annotations["info"]["indx"]])
        sorted_indices = np.argsort(start_indices)

        sorted_info_index = list(np.array(self.annotations["info"]["indx"])[sorted_indices])
        sorted_annos = list(np.array(self.annotations["language"]["task"])[sorted_indices])

        key_state_indices = [sorted_info_index[i][1] for i in range(start_seq_index, start_seq_index + n_seqs)]

        start_index = sorted_info_index[0][0]
        key_state_indices_local = [curr_index - sorted_info_index[0][1] for curr_index in key_state_indices]
        end_index = max(key_state_indices)

        annos_seqs = sorted_annos[start_seq_index: start_seq_index + n_seqs]

        obs_seqs = []
        for index in range(start_index, end_index):
            try:
                frame = np.load(os.path.join(self.dataset_path, f"episode_{index:07d}.npz"), allow_pickle=True)
                obs_seqs.append(dict(frame))
            except FileNotFoundError as e:
                print(e)
                pass
        
        obs_seqs = {key: [dic[key] for dic in obs_seqs] for key in obs_seqs[0]}

        return obs_seqs, annos_seqs, key_state_indices_local
