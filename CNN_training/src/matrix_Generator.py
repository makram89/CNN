# USAGE
# python new_tester.py --model AUG.model -L labels.txt --image examples/
# python CNN_training\src\new_tester.py -L CNN_training\labels.txt -m CNN_training\AUG.model -i examples\plane.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os
from CNN_training.src.Dataset import Dataset
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                help="path to trained model model")

args = vars(ap.parse_args())

DATASET_PATH = "E:\IT\CNN\dataset"
LABELS_PATH = "E:\IT\CNN\CNN_training\labels.txt"
VAL_PATH = "E:\IT\CNN\_val_set"
dataSet = Dataset(DATASET_PATH, LABELS_PATH, VAL_PATH)
dataSet.load_only_images(dataset_path=VAL_PATH)
validation_matrix = []

print("[LOG] loading network...")
model = load_model(args["model"])

file = open(LABELS_PATH, 'r')
classes = [x[0:-1] for x in file]
# print(classes)

for i in range(len(classes) + 1):
    if i == 0:
        validation_matrix.append(["RANKING"])
        [validation_matrix[0].append(x) for x in classes]
    else:
        validation_matrix.append([classes[i - 1]])
        [validation_matrix[i].append(0) for x in range(len(classes))]

for x in validation_matrix:
    print(x)

for x in range(len(dataSet.validation_data)):
    image = cv2.imread(dataSet.validation_data_images[x])
    # cv2.imshow("ops", imutils.resize(image, 450))
    # cv2.waitKey(0)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    result = model.predict(image)[0]
    idx = np.argmax(result)
    # print(idx)
    id_origin = classes.index(dataSet.val_labels_txt[x])

    validation_matrix[id_origin + 1][idx + 1] += 1

for x in validation_matrix:
    print(x)

fig, axs = plt.subplots()
axs.set_title(args["model"].split('\\')[-1], loc='center')
axs.axis('tight')
axs.axis('off')
table = axs.table(cellText=validation_matrix[1:], loc='center', colLabels=validation_matrix[0])
plt.show()

# # classify the input image
# print("[LOG] classifying image...")
#
#
# idx = np.argmax(result)
#
# label_class = classes[idx]
#

#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
