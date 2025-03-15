import logging
from pathlib import Path
import sys
import threading
from collections import OrderedDict
import pickle
import torch

sys.path.append(str(Path(__file__).absolute().parents[2]))
from mode.models.perceptual_encoders.pretrained_resnets import FiLMLayer



class AdvancedVisLangEmbeddingBuffer:
    def __init__(self, vision_encoder, language_encoder, lang_goal_buffer_size=10000):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.lang_goal_buffer_size = lang_goal_buffer_size

        self.lang_goal_buffer = OrderedDict()
        self.buffer_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)


    def get_or_encode_vis_lang_batch(self, images: torch.Tensor, texts: list):
        try:
            with self.buffer_lock:
                uncached_texts = [text for text in texts if text not in self.lang_goal_buffer]

            if uncached_texts:
                encoded_lang_batch = self.language_encoder(uncached_texts)
                
                for uncached_text, encoded_text in zip(uncached_texts, encoded_lang_batch):
                    self.add_to_lang_buffer(key=uncached_text, value=encoded_text)
            
            with self.buffer_lock:
                encoded_texts = torch.stack([self.lang_goal_buffer[text] for text in texts]).squeeze(1)

        except Exception as e:
            self.logger.warning(f"Error with buffer while encoding texts: '{e}'. Encoding without buffer.")

            # If an error occurs, encode text batch from scratch
            encoded_texts = self.language_encoder(texts).squeeze(1)

        return encoded_texts


    def add_to_lang_buffer(self, key, value):
        with self.buffer_lock:
            if len(self.lang_goal_buffer) >= self.lang_goal_buffer_size:
                self.lang_goal_buffer.popitem(last=False)
            self.lang_goal_buffer[key] = value


    def get_vis_lang_goal_embedding(self, vis_goal, lang_goal):
        return self.get_or_encode_vis_lang_batch(torch.tensor(vis_goal).unsqueeze(0), [lang_goal])


    def get_vis_lang_goal_embeddings(self, vis_goals, lang_goals):
        return self.get_or_encode_vis_lang_batch(vis_goals, lang_goals)


    def clear_lang_buffer(self):
        with self.buffer_lock:
            self.lang_goal_buffer.clear()


    def get_lang_buffer_size(self):
        with self.lang_buffer_lock:
            return len(self.lang_goal_buffer)


    def save_lang_buffer(self, filepath):
        with self.lang_buffer_lock:
            with open(filepath, 'wb') as f:
                pickle.dump(self.lang_goal_buffer, f)


    def load_lang_buffer(self, filepath):
        with open(filepath, 'rb') as f:
            loaded_buffer = pickle.load(f)
        with self.lang_buffer_lock:
            self.lang_goal_buffer = OrderedDict(list(loaded_buffer.items())[-self.lang_goal_buffer_size:])
