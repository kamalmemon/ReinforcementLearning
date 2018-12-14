import cv2
import numpy as np
from PIL import Image

def prepro(frame):
     print(frame.shape)
     """ prepro 210x210x3 uint8 frame into 6400 (80x80) 1D float vector """
     im = Image.fromarray(frame)
     #im = im.convert('L'); # convert to gray scale
     frame = np.asarray(im).copy() # copy to numpy array
     #frame = frame[34:34 + 210, :210]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
     frame = cv2.resize(frame, (105, 105))
     frame = cv2.resize(frame, (42, 42))
     frame = frame.mean(2, keepdims=True)
     frame = frame.astype(np.float32)
     frame *= (1.0 / 255.0)
     frame = np.moveaxis(frame, -1, 0)
     return frame

class ObsNorm():
     def __init__(self, env=None):
          self.state_mean = 0
          self.state_std = 0
          self.alpha = 0.9999
          self.num_steps = 0

     def prepro(self, frame):
          return self.normalize(prepro(frame))
          
     def reset(self):
          self.state_mean = 0
          self.state_std = 0
          self.alpha = 0.9999
          self.num_steps = 0
          
     def normalize(self, observation):
          self.num_steps += 1
          self.state_mean = self.state_mean * self.alpha + \
          observation.mean() * (1 - self.alpha)
          self.state_std = self.state_std * self.alpha + \
          observation.std() * (1 - self.alpha)
          unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
          unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
          
          return (observation - unbiased_mean) / (unbiased_std + 1e-8)
