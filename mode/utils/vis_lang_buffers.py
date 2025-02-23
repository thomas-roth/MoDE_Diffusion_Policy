import threading
from collections import OrderedDict
import pickle
import torch
from PIL import Image

class AdvancedVisLangEmbeddingBuffers:
    def __init__(self, vision_encoder, language_encoder, vis_goal_buffer_size=1000, lang_goal_buffer_size=10000):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.vis_goal_buffer_size = vis_goal_buffer_size
        self.lang_goal_buffer_size = lang_goal_buffer_size
        self.vis_goal_buffer = OrderedDict()
        self.lang_goal_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()

    def get_or_encode_vis_lang_batch(self, images, texts):
        # print(images)
        # print(texts)

        if isinstance(texts, str):
            texts = [texts]

        try:
            with self.buffer_lock:
                uncached_images = [image for image in images if image not in self.vis_goal_buffer]
                uncached_texts = [text for text in texts if text not in self.lang_goal_buffer]
            
            if uncached_images:
                preprocessed_images = [self.preprocess_image(image) for image in uncached_images]
                encoded_vis_batch = self.vision_encoder(preprocessed_images)
            
                for image, embedding in zip(uncached_images, encoded_vis_batch):
                    self.add_to_vis_buffer(image, embedding)

            if uncached_texts:
                encoded_lang_batch = self.language_encoder(uncached_texts)
                
                for text, embedding in zip(uncached_texts, encoded_lang_batch):
                    self.add_to_lang_buffer(text, embedding)
            
            with self.buffer_lock:
                encoded_images = [self.vis_goal_buffer[image] for image in images]
                encoded_texts = [self.lang_goal_buffer[text] for text in texts]
            
            encoded_goal = [torch.cat((encoded_image, encoded_text)) for encoded_image, encoded_text in zip(encoded_images, encoded_texts)]
            return torch.stack(encoded_goal)

        except Exception as e:
            print(f"Error encoding images and texts: {e}")
            # If all else fails, return dummy tensors
            # Assuming the output dimensions of the vision and language encoders are known
            return torch.zeros((len(images), self.vision_encoder.output_dim)), torch.zeros((len(texts), self.language_encoder.output_dim))
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        preprocessed_image = self.vision_encoder.preprocess_image(image)
        return preprocessed_image

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
