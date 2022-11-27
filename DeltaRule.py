import numpy as np
from PIL import Image
import os


class DeltaRule:
    def __init__(self):
        self.alphabet_image = os.listdir('./alphabet')
        self.weights = np.random.random((len(self.alphabet_image), 49))
        self.target = np.array([128, 129, 130, 131, 132, 133, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146,
                                147, 148, 149, 150, 151])
        self.epsilon = 0.1
        self.t = 0
        self.x0 = -1
        self.learning_rate = 0.000001  # 0.00001
        self.y_pred = np.zeros([len(self.alphabet_image)])

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
                for j, b in zip(range(0, len(img_as_array), 3), range(0, len(self.weights[i]))):
                    if img_as_array[j] == 0 or img_as_array[j] == 1:
                        img_as_array[j] = 1
                    elif img_as_array[j] == 254 or img_as_array[j] == 255:
                        img_as_array[j] = 0
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
            if count == 49:
                print('COUNT = 49')
                let = self.calculating_weights(i, img_as_array)
                print(let)
                alphabet = {128: 'А', 129: 'Б', 130: 'В', 131: 'Г', 132: 'Д', 133: 'Е', 135: 'Ж',
                            136: 'З', 137: 'И', 139: 'К', 140: 'Л', 141: 'М', 142: 'Н', 143: 'О', 144: 'П', 145: 'Р',
                            146: 'С', 147: 'Т', 148: 'У', 149: 'Ф', 150: 'Х', 151: 'Ц'}
                for ch in alphabet.keys():
                    if ch == let:
                        return alphabet[ch]
