import numpy as np
from PIL import Image
import os


class DeltaRule:
    def __init__(self):
        self.alphabet_image = os.listdir('./alphabet')
        self.weights = np.random.random((len(self.alphabet_image), 35))
        self.target = np.array([128, 129, 130, 131, 132, 133])
        self.epsilon = 0.1
        self.t = 0
        self.x0 = -1
        self.learning_rate = 1
        self.y_pred = np.array([])


    def Relu(self, NET: int):
        if NET > 0:
            return NET
        elif NET <= 0:
            return 0

    def calculating_weights(self, i: int, img_as_array):
        Sum = 0.0
        for j, b in zip(range(0, len(img_as_array), 3), range(len(self.weights[i]))):
            Sum += np.multiply(img_as_array[j], self.weights[i][b])
        Sum -= self.t
        return self.Relu(round(Sum))

    def train(self):
        for i in range(0, len(self.integer_image)):
            img_as_array = np.array([])
            with Image.open('./digit/' + str(self.integer_image[i])) as img:
                img_as_array = np.append(img_as_array, img)
                for j in range(len(img_as_array)):
                    if img_as_array[j] == 0 or img_as_array[j] == 1:
                        img_as_array[j] = 0.1
                    elif img_as_array[j] == 254 or img_as_array[j] == 255:
                        img_as_array[j] = 0

            while self.calculating_weights(i, img_as_array) != self.target[i]:
                for b in range(len(self.weights[i])):
                    self.weights[i][b] = self.weights[i][b] + self.learning_rate * (self.target[i] - self.y_pred[i])
                self.t = self.t + self.learning_rate * (self.target[i] - self.y_pred[i]) * self.x0
