import numpy as np
from PIL import Image
import os


class HebbMethod:
    def __init__(self):
        self.integer_image = os.listdir('./digit')
        self.weights = np.zeros(shape=(len(self.integer_image), 15))
        # self.y = np.zeros(shape=(len(self.integer_image), len(self.integer_image)))
        # self.img_as_array = np.array([])

    def Relu(self, NET: int):
        if NET > 0:
            return NET
        elif NET <= 0:
            return 0

    def calculating_weights(self, i: int, img_as_array):
        Sum = 0.0
        for j, b in zip(range(0, len(img_as_array), 3), range(len(self.weights[i]))):
            Sum += np.multiply(img_as_array[j], self.weights[i][b])
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

            while self.calculating_weights(i, img_as_array) != i:
                for j, b in zip(range(0, len(img_as_array), 3), range(len(self.weights[i]))):
                    self.weights[i][b] = self.weights[i][b] + (np.multiply(img_as_array[j], i))
                    print('self.weights[' + str(i) + '][' + str(b) + '] = ' + str(self.weights[i][b]))

    def predict(self, img_as_array):
        for i in range(0, len(self.integer_image)):
            count = 0
            for j, b in zip(range(0, len(img_as_array), 3), range(0, len(self.weights[i]))):
                if img_as_array[j] > 0 and self.weights[i][b] > 0:
                    count += 1
                elif img_as_array[j] == 0 and self.weights[i][b] == 0:
                    count += 1
                elif img_as_array[j] == 0 and self.weights[i][b] != 0:
                    break
                if img_as_array[j] > 0 and self.weights[i][b] == 0:
                    break
            if count == 15:
                return self.calculating_weights(i, img_as_array)
