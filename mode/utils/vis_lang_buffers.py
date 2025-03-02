from pathlib import Path
import sys
import threading
from collections import OrderedDict
import pickle
import pyhash
import torch

sys.path.append(str(Path(__file__).absolute().parents[2]))
from mode.models.perceptual_encoders.pretrained_resnets import FiLMLayer



class AdvancedVisLangEmbeddingBuffers:
    def __init__(self, vision_encoder, language_encoder, vis_goal_buffer_size=1000, lang_goal_buffer_size=10000):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.vis_goal_buffer_size = vis_goal_buffer_size
        self.lang_goal_buffer_size = lang_goal_buffer_size
        self.vis_goal_buffer = OrderedDict()
        self.lang_goal_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()

        self.hasher = pyhash.fnv1a_64()
        self.goal_projection_layer = FiLMLayer(condition_dim=self.language_encoder.output_dim, num_features=512)

    def get_or_encode_vis_lang_batch(self, images, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            with self.buffer_lock:
                uncached_images = [image for image in images if self._hash_tensor(image) not in self.vis_goal_buffer]
                uncached_texts = [text for text in texts if text not in self.lang_goal_buffer]
            
            if uncached_images:
                encoded_vis_batch = self.vision_encoder(uncached_images)
            
                for uncached_image, embedding in zip(uncached_images, encoded_vis_batch):
                    self.add_to_vis_buffer(self._hash_tensor(uncached_image), value=embedding)

            if uncached_texts:
                encoded_lang_batch = self.language_encoder(uncached_texts)
                
                for text, embedding in zip(uncached_texts, encoded_lang_batch):
                    self.add_to_lang_buffer(key=text, value=embedding)
            
            with self.buffer_lock:
                encoded_images = torch.stack([self.vis_goal_buffer[self._hash_tensor(image)] for image in images]).squeeze()
                encoded_texts = torch.stack([self.lang_goal_buffer[text] for text in texts]).squeeze()

            encoded_goal = self.goal_projection_layer(x=encoded_images, condition=encoded_texts, unsqueeze=False)
            return encoded_goal

        except Exception as e:
            print(f"Error encoding images and texts: key {e} not found in buffer. Encoding from scratch.")

            # If an error occurs, encode the batch from scratch
            encoded_vis_batch = self.vision_encoder(images).squeeze()
            encoded_lang_batch = self.language_encoder(texts).squeeze()

            encoded_goal = self.goal_projection_layer(x=encoded_vis_batch, condition=encoded_lang_batch, unsqueeze=False)
            return encoded_goal

    def _hash_tensor(self, tensor):
        # required to avoid using mutable tensors as keys in self.vis_goal_buffer
        # FIXME: in some cases keys not found in buffer => change hasher? change whole approach of transforming images into immutable representations?
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return self.hasher(tensor_bytes)

    def add_to_lang_buffer(self, key, value):
        with self.buffer_lock:
            if len(self.lang_goal_buffer) >= self.vis_goal_buffer_size:
                self.lang_goal_buffer.popitem(last=False)
            self.lang_goal_buffer[key] = value

    def add_to_vis_buffer(self, key, value):
        with self.buffer_lock:
            if len(self.vis_goal_buffer) >= self.vis_goal_buffer_size:
                self.vis_goal_buffer.popitem(last=False)
            self.vis_goal_buffer[key] = value

    def get_vis_lang_goal_embedding(self, vis_goal, lang_goal):
        return self.get_or_encode_vis_lang_batch([vis_goal], [lang_goal])

    def get_vis_lang_goal_embeddings(self, vis_goals, lang_goals):
        return self.get_or_encode_vis_lang_batch(vis_goals, lang_goals)

    def clear_vis_buffer(self):
        with self.buffer_lock:
            self.vis_goal_buffer.clear()

    def clear_lang_buffer(self):
        with self.buffer_lock:
            self.lang_goal_buffer.clear()

    def get_vis_buffer_size(self):
        with self.buffer_lock:
            return len(self.vis_goal_buffer)

    def get_lang_buffer_size(self):
        with self.lang_buffer_lock:
            return len(self.lang_goal_buffer)

    def save_vis_buffer(self, filepath):
        with self.buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.vis_goal_buffer, f)

    def save_lang_buffer(self, filepath):
        with self.lang_buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.lang_goal_buffer, f)

    def load_vis_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.buffer_lock:
            self.vis_goal_buffer = OrderedDict(list(loaded_buffer.items())[-self.vis_goal_buffer_size:])

    def load_lang_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.lang_buffer_lock:
            self.lang_goal_buffer = OrderedDict(list(loaded_buffer.items())[-self.lang_goal_buffer_size:])
