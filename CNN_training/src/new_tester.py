# USAGE
# python tester.py --model water.model --labelbin lb.pickle --image examples/

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")

ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-L", required=True,
                help="path to input labels")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
output = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[LOG] loading network...")
model = load_model(args["model"])
file = open(args["L"], 'r')
classes = [x[0:-1] for x in file]
print(classes)

# classify the input image
print("[LOG] classifying image...")
result = model.predict(image)[0]
idx = np.argmax(result)
print(idx)
label_class = classes[idx]

# build the label and draw the label on the image
label = " {} {:.2f}% ".format(label_class, result[idx] * 100) + ""

labels = [" {} {:.2f}% ".format(label_class[i], result[i] * 100) + "/n " for i in range(len(result))]

output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

output2 = output.copy()
cv2.putText(output2, labels, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.imshow("Output2", output2)
cv2.waitKey(0)
