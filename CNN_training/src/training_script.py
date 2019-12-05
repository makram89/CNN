# python train.py --dataset dataset --model {name}.model --labelbin lb.pickle

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from CNN_training.src.Dataset import Dataset
from CNN_training.nnetwork.smallervggnet import SmallerVGGNet
import tensorflow as tf
import keras as k
import matplotlib.pyplot as plt
# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")


class Trainer:
    EPOCHS = 500
    INIT_LR = 1e-3
    BS = 32
    model = 0
    initial_epoch = 0

    def __init__(self, dataset):
        print("[LOG] Creating model...")
        self.dataset = dataset

        self.opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)

    def create_model(self):
        self.model = SmallerVGGNet.build(width=self.dataset.IMAGE_DIMS[1], height=self.dataset.IMAGE_DIMS[0],
                                         depth=self.dataset.IMAGE_DIMS[2],
                                         classes=len(self.dataset.classes))

        # COMPILING MODEL
        print("[LOG] Compiling model... ")
        self.model.compile(loss="categorical_crossentropy", optimizer=self.opt, metrics=['accuracy'])
        print("[LOG] Ready model")

    def load_(self, model_path):
        self.model = k.models.load_model(model_path)
        file = open("ep.txt", "r")
        self.initial_epoch = int(file.read())
        file.close()
        print(self.initial_epoch)

    def training_augmented(self):
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")

        print("[LOG] training AUG")
        net = self.model.fit_generator(
            aug.flow(self.dataset.data, self.dataset.labels, batch_size=self.BS),
            validation_data=(self.dataset.validation_data, self.dataset.val_labels),
            steps_per_epoch=len(self.dataset.data) // self.BS,
            epochs=self.initial_epoch + self.EPOCHS, initial_epoch=self.initial_epoch
        )

        # TODO: if lepszy od modelu to zapisz
        print("[LOG] compare")

        print("[LOG] saving")

        file = open("ep.txt", "w")
        file.write(str(self.initial_epoch + self.EPOCHS))
        file.close()

        self.model.save("AUG.model")
        self.plot_it(net)

        print("[LOG] Ploted")

    def standard_training(self):
        print("[LOG] training STD")
        net = self.model.fit(self.dataset.data, self.dataset.labels,
                             epochs=self.EPOCHS, batch_size=self.BS)
        print("[LOG] saving")
        self.model.save("STD.model")
        self.plot_it(net)

    def custom_training(self):
        print("[LOG] training CUS")
        net = self.model.fit(self.dataset.data, self.dataset.labels,
                             epochs=self.EPOCHS, batch_size=self.BS)

        test_loss, test_acc = self.model.evaluate(self.dataset.validation_data, self.dataset.val_labels)

        print("[LOG] saving")
        self.model.save("CUS.model")

    def test_it(self):
        pass

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
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig("plot.png")


DATASET_PATH = "E:\IT\CNN\dataset"
LABELS_PATH = "E:\IT\CNN\CNN_training\labels.txt"
VAL_PATH = "E:\IT\CNN\_val_set"
dataSet = Dataset(DATASET_PATH, LABELS_PATH, VAL_PATH)

trainer = Trainer(dataSet)
# trainer.create_model()
trainer.load_("AUG.model")
trainer.training_augmented()
