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
        self.learning_rate = 0.00001
        self.y_pred = np.zeros([6])

    def Relu(self, i, NET: int):
        if NET > 0:
            self.y_pred[i] = NET
            return NET
        elif NET <= 0:
            return 0

    def calculating_weights(self, i: int, img_as_array):
        Sum = 0.0
        for j, b in zip(range(0, len(img_as_array), 3), range(len(self.weights[i]))):
            Sum += np.multiply(img_as_array[j], self.weights[i][b])
        Sum -= self.t
        return self.Relu(i, round(Sum))

    def train(self):
        for i in range(0, len(self.alphabet_image)):
            img_as_array = np.array([])
            with Image.open('./alphabet/' + str(self.alphabet_image[i])) as img:
                img_as_array = np.append(img_as_array, img)
                # print(self.weights[i])
                for j, b in zip(range(0, len(img_as_array), 3), range(0, len(self.weights[i]))):
                    if img_as_array[j] == 0 or img_as_array[j] == 1:
                        img_as_array[j] = 1
                    elif img_as_array[j] == 254 or img_as_array[j] == 255:
                        img_as_array[j] = 0
                        #print(self.weights[i][b] )
                        self.weights[i][b] = 0

            while (self.target[i] - self.calculating_weights(i, img_as_array)) > self.epsilon:
                for j, b in zip(range(0, len(img_as_array), 3), range(0, len(self.weights[i]))):
                    self.weights[i][b] = self.weights[i][b] + self.learning_rate * (self.target[i] - self.y_pred[i]) * \
                                         img_as_array[j]
                self.t = self.t + self.learning_rate * (self.target[i] - self.y_pred[i]) * self.x0

    def predict(self, img_as_array):
        for i in range(0, len(self.alphabet_image)):
            count = 0
            for j, b in zip(range(0, len(img_as_array), 3), range(0, len(self.weights[i]))):
                if img_as_array[j] > 0 and self.weights[i][b] > 0:
                    count += 1
                elif img_as_array[j] == 0 and self.weights[i][b] == 0:
                    count += 1
                elif img_as_array[j] == 0 and self.weights[i][b] > 0:
                    break
                if img_as_array[j] > 0 and self.weights[i][b] == 0:
                    break
            if count == 35:
                # print('self.calculating_weights(i, img_as_array) = ' + str(self.calculating_weights(i, img_as_array)))
                let = int(self.calculating_weights(i, img_as_array))
                alphabet = {128: 'А', 129: 'Б', 130: 'В', 131: 'Г', 132: 'Д', 133: 'Е', 134: 'Ё', 135: 'Ж'}
                for ch in alphabet.keys():
                    if ch == let:
                        return alphabet[ch]
