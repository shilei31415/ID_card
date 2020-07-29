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
        self.image_names = os.listdir(path+"ANN/ID_data")
        for image_name in self.image_names:
            self.image_path.append(path + "ANN/ID_data" +'/' + image_name)
        cv2.namedWindow("yuantu", cv2.WINDOW_NORMAL)
        self.nums_path = []
        for i in range(12):
            self.num_path = []
            num_path = os.listdir(path+str(i))
            for num_name in num_path:
                self.num_path.append(path+str(i)+'/'+num_name)
            self.nums_path.append(self.num_path)
        # print(self.nums_path[10][10])

    def make_data(self):
        image = None
        label = np.zeros((18, 12))
        for i in range(18):
            j = random.randint(0, 11)
            label[i, j] = 1
            path = random.choice(self.nums_path[j])
            img = cv2.imread(path)
            h, w, c = img.shape
            img = cv2.resize(img, (int(40/h*w), 40))
            if image is None:
                image=img
            else:
                image = np.hstack((image, img))
        image = cv2.resize(image, (448, 40))
        # print(label)
        cv2.imshow("make_data", image)
        cv2.waitKey(0)
        image = image.astype(np.float)

        k = (np.random.random((image_rows, image_cols, 1)) - 0.5) * 0.2 + 1
        image = image * k
        # print(image)
        image[image > 255] = 255
        image[image < 0] = 0

        image = image / 255.0
        return image, label



    def image(self, index):
        image = cv2.imread(self.image_path[index])
        # print(self.image_path[index])
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
            if random.random()<0.2:
                index = random.randint(0, len(self.image_names)-1)
                images.append(self.image(index))
                labels.append(self.label(index))
            else:
                image, label = self.make_data()
                images.append(image)
                labels.append(label)
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
    data = data_base("/home/shilei/CLionProjects/ID_card/")
    for i in range(100):
        images, labels = data.get_data(10)
    print(images.shape)
    print(labels.shape)
