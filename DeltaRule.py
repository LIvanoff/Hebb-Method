import numpy as np
from PIL import Image
import os

class DeltaRule:
    def __init__(self):
        self.alphabet_image = os.listdir('./alphabet')
        self.T = 0.0
        self.weights = np.zeros(shape=(len(self.alphabet_image), 35))
        self.img_as_array = np.array([])