import numpy as np
from PIL import Image
import os

class HebbMethod:
    def __init__(self):
        self.integer_image = os.listdir('./integer')
        self.T = 0.0
        self.weights = np.zeros(shape=(len(self.integer_image), 15))
        # self.img_as_array = np.array([])

    def Relu(self, NET: int):
        if NET > 0:
            return NET
        elif NET <= 0:
            return 0

    def calculating_weights(self, i: int, img_as_array):
        Sum = 0.0
        for j, b in zip(range(0, len(img_as_array), 3), range(0, 15, 1)):
            Sum += np.multiply(img_as_array[j], self.weights[i][b])
        return self.Relu(round(Sum))

    def train(self):
        for i in range(0,len(self.integer_image)):
            img_as_array = np.array([])
            with Image.open('./integer/' + str(self.integer_image[i])) as img:
                img_as_array = np.append(img_as_array, img)
                for j in range(len(img_as_array)):
                    if img_as_array[j] == 0 or img_as_array[j] == 1:
                        img_as_array[j] = 0.1
                    elif img_as_array[j] == 254 or img_as_array[j] == 255:
                        img_as_array[j] = 0

            Sum = self.calculating_weights(i, img_as_array)  # - self.T

            while ((Sum) != i):
                for b, j in zip(range(0, len(img_as_array), 3), range(len(self.weights[i]))):
                    #print('weights['+str(i)+']['+str(j)+'] = '+str(self.weights[i][j]))
                    self.weights[i][j] = self.weights[i][j] + np.multiply(img_as_array[b], i)
                    #print('new weights[' + str(i) + '][' + str(j) + '] = ' + str(self.weights[i][j]))
                # self.T = self.T - i
                Sum = self.calculating_weights(i, img_as_array)  # - self.T

    def predict(self, img_as_array):
        num = 0
        for i in range(0,len(self.integer_image)):
            count = 0
            for j, b in zip(range(0, len(img_as_array), 3), range(0, 15, 1)):
                if img_as_array[j] > 0 and self.weights[i][b] > 0:
                    count += 1
                elif img_as_array[j] == 0 and self.weights[i][b] == 0:
                    count += 1
                elif img_as_array[j] == 0 and self.weights[i][b] != 0:
                    break
                if img_as_array[j] > 0 and self.weights[i][b] == 0:
                    break
            if count == 15:
                num = i
        return self.calculating_weights(num, img_as_array)


