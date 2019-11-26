from imutils import paths
import cv2
from keras_preprocessing.image import img_to_array
import os
import numpy as np

class Dataset:
    data = []
    validation_data = []
    labels = []
    val_labels = []
    IMAGE_DIMS = (96, 96, 3)

    def __init__(self, dataset_path, label_path, val_path):
        print("[LOG] load data from files ...")
        # self.load_labels(label_path)
        self.data, self.labels = self.load_images(dataset_path)
        self.validation_data, self.val_labels = self.load_images(val_path)
        self.scale_data()
        print("[LOG] Loaded")

    def load_images(self, dataset_path):
        temp_data = []
        temp_labels = []
        imagePaths = sorted(list(paths.list_images(dataset_path)))
        for imagePath in imagePaths:
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (self.IMAGE_DIMS[1], self.IMAGE_DIMS[0]))
            image = img_to_array(image)
            temp_data.append(image)
            temp_labels.append(imagePath.split(os.path.sep)[-2])
        return temp_data, temp_labels

    # zmiana wartości na takie z zaresu 0-1
    def scale_data(self):
        self.data = np.array(self.data, dtype="float") / 255.0
        self.validation_data = np.array(self.data, dtype="float") / 255.0

    def load_labels(self, label_path):
        file = open(label_path, "r")
        [print(x) for x in file]
        file.close()

    def create_labels(self):
        file = open("labels.txt", "w")
        for x in np.unique(self.labels):
            file.write(x + "\n")
        file.close()

    def check_ok(self):
        [print(x) for x in self.labels]