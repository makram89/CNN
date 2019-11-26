# python train.py --dataset dataset --model {name}.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background

import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from CNN_training.Dataset import Dataset
from CNN_training.nnetwork.smallervggnet import SmallerVGGNet

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


class Trainer:
    EPOCHS = 10
    INIT_LR = 1e-3
    BS = 32

    def __init__(self, dataset):
        print("[LOG] Creating model...")
        self.dataset = dataset
        self.aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                      height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                      horizontal_flip=True, fill_mode="nearest")
        # self.opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

        self.model = SmallerVGGNet.build(width=self.dataset.IMAGE_DIMS[1], height=self.dataset.IMAGE_DIMS[0],
                                         depth=self.dataset.IMAGE_DIMS[2], classes=len(np.unique(self.dataset.labels)))
        # COMPILING MODEL
        print("[LOG] Compiling model... ")
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        print("[LOG] Ready model")

    def training_augmented(self):
        print("[LOG] training AUG")
        net = self.model.fit_generator(self.aug.flow(self.dataset.data, self.dataset.labels, batch_size=self.BS),
                                       validation_data=(self.dataset.validation_data, self.dataset.val_labels),
                                       steps_per_epoch=len(self.dataset.data) // self.BS,
                                       epochs=self.EPOCHS, verbose=1)
        print("[LOG] saving")
        self.model.save("AUG.model")
        self.plot_it(net)

    def plot_it(self, net):
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        N = self.EPOCHS
        plt.plot(np.arange(0, N), net.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), net.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), net.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), net.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig("plot.png")


DATASET_PATH = "E:\IT\CNN\dataset"
LABELS_PATH = "E:\IT\CNN\CNN_training\labels.txt"
VAL_PATH = "E:\IT\CNN\_val_set"
dataSet = Dataset(DATASET_PATH, LABELS_PATH, VAL_PATH)

trainer = Trainer(dataSet)
trainer.training_augmented()
