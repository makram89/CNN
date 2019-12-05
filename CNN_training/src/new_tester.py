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
# print(result)
idx = np.argmax(result)
# print(idx)
# print(classes[idx])
# print(result[idx])
label_class = classes[idx]

# build the label and draw the label on the image
label = " {} {:.6f}% ".format(label_class, result[idx] * 100) + ""

labels = [" {} {:.6f}x1000 ".format(classes[i], result[i] * 1000) for i in range(len(result))]
# labels = '; \n '.join(labels)

output = imutils.resize(output, width=480)
output2 = output.copy()

cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
for x in range(len(labels)):
    cv2.putText(output2, labels[x], (50, x*20+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 0), 1)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.imshow("Output2", output2)
cv2.waitKey(0)
cv2.destroyAllWindows()
