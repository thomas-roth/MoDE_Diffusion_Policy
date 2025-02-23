import threading
from collections import OrderedDict
import pickle
import torch
from PIL import Image

class AdvancedVisEmbeddingBuffer:
    def __init__(self, vision_encoder, vis_goal_buffer_size=1000):
        self.vision_encoder = vision_encoder
        self.vis_goal_buffer_size = vis_goal_buffer_size
        self.vis_goal_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()

    def get_or_encode_batch(self, images):
        # print(images)

        try:
            with self.buffer_lock:
                uncached_images = [image for image in images if image not in self.vis_goal_buffer]
            
            if uncached_images:
                # Load and preprocess images
                preprocessed_images = [self.preprocess_image(image) for image in uncached_images]
                
                # Encode preprocessed
                encoded_batch = self.vision_encoder(preprocessed_images)
                
                for image, embedding in zip(uncached_images, encoded_batch):
                    self.add_to_buffer(image, embedding)
            
            with self.buffer_lock:
                encoded_images = [self.vis_goal_buffer[image] for image in images]
            
            return torch.stack(encoded_images)

        except Exception as e:
            print(f"Error encoding images: {e}")
            # If all else fails, return a dummy tensor
            # Assuming the output dimension of the vision encoder is known
            return torch.zeros((len(images), self.vision_encoder.output_dim))
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = self.vision_encoder.preprocess_image(image)
        return preprocessed_image

    def add_to_buffer(self, key, value):
        with self.buffer_lock:
            if len(self.vis_goal_buffer) >= self.vis_goal_buffer_size:
                self.vis_goal_buffer.popitem(last=False)
            self.vis_goal_buffer[key] = value

    def get_vis_goal_embedding(self, vis_goal):
        return self.get_or_encode_batch([vis_goal])

    def get_vis_goal_embeddings(self, vis_goals):
        return self.get_or_encode_batch(vis_goals)

    def clear_buffer(self):
        with self.buffer_lock:
            self.vis_goal_buffer.clear()

    def get_buffer_size(self):
        with self.buffer_lock:
            return len(self.vis_goal_buffer)

    def save_buffer(self, filepath):
        with self.buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.vis_goal_buffer, f)

    def load_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.buffer_lock:
            self.vis_goal_buffer = OrderedDict(list(loaded_buffer.items())[-self.vis_goal_buffer_size:])
