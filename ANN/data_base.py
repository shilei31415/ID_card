import cv2
import os
import random
import numpy as np

ID_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# for model
# image_rows = 75
# image_cols = 724

# for model2
image_rows = 40
image_cols = 448

class data_base:
    def __init__(self, path):
        self.floder_path = path
        self.image_path = []
        self.image_names = os.listdir(path)
        for image_name in self.image_names:
            self.image_path.append(path + '/' + image_name)
        cv2.namedWindow("yuantu", cv2.WINDOW_NORMAL)

    def image(self, index):
        image = cv2.imread(self.image_path[index])
        image = cv2.resize(image, (image_cols, image_rows))
        cv2.imshow("yuantu", image)
        cv2.waitKey(1)
        image = image.astype(np.float)
        # print(image)

        k = (np.random.random((image_rows, image_cols, 1))-0.5)*0.2+1
        image = image*k
        # print(image)
        image[image>255]=255
        image[image<0]=0
        # print(image)

        image = image/255.0
        # print(image)
        return image

    def label(self, index):
        name = self.image_names[index][0:18]
        label = np.zeros((18, 12))
        for i in range(0, 18):
            if name[i] in ID_num:
                j = ID_num.index(name[i])
                label[i, j] = 1
            elif name[i] == 'x' or name[i] == 'X':
                label[i, 10] = 1
            else:
                label[i, 11] = 1

        # print(name)
        # print(label)
        return label

    def get_data(self, batch):
        images = []
        labels = []
        for i in range(batch):
            index = random.randint(0, len(self.image_names)-1)
            images.append(self.image(index))
            labels.append(self.label(index))
        images = np.array(images)
        labels = np.array(labels)
        return images, labels

    def test_image(self):
        index = random.randint(0, len(self.image_names) - 1)
        image = cv2.imread(self.image_path[index])
        image = cv2.resize(image, (image_cols, image_rows))
        cv2.imshow("yuantu", image)
        cv2.waitKey(1)
        return image


if __name__ == '__main__':
    data = data_base("/home/shilei/CLionProjects/ID_card/ANN/ID_data")
    for i in range(100):
        images, labels = data.get_data(10)
    print(images.shape)
    print(labels.shape)
