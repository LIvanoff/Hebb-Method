from HebbMethod import HebbMethod
import numpy as np
from PIL import Image
import os

if __name__ == "__main__":
    hm = HebbMethod()
    hm.train()
    #print(hm.weights[1])
    integer_image = os.listdir('./integer')
    img_as_array = np.array([])
    with Image.open('./integer/цифра 4.jpg') as img:
        img_as_array = np.append(img_as_array, img)
        for j in range(len(img_as_array)):
            if img_as_array[j] == 0 or img_as_array[j] == 1:
                img_as_array[j] = 0.1
            elif img_as_array[j] == 254 or img_as_array[j] == 255:
                img_as_array[j] = 0

    print('predict: ' + str(hm.predict(img_as_array)))