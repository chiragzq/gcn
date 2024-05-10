from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.utils import image_dataset_from_directory

import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import pandas as pd
import numpy as np

class GeoCnn:
    def __init__(self):
        self.image_width = 640#1280//2
        self.image_height = 295#640//2
        self.num_countries = 35
        self.input_shape = (self.image_height, self.image_width, 3)
    
    def setup_model(self):
        self.model = Sequential()
        self.model.add(ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape))
        for layer in self.model.layers:
          layer.trainable = False
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_countries, activation="softmax"))
        self.model.compile(optimizer=Adam(0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def load_dataset(self):
        # load classes
        self.csv_data = pd.read_csv("dataset/coords_country.csv")
        self.country_classes = self.csv_data["country"].unique()
        self.output_labels = self.csv_data["country"].tolist()
        self.output_labels = [np.where(self.country_classes == x)[0][0] for x in self.output_labels]
        print(len(self.output_labels))
        print(max(self.output_labels))
        self.train_data, self.test_data = image_dataset_from_directory(
            "dataset/img_low", labels=self.output_labels, image_size=(self.image_height, self.image_width), batch_size=32,
            validation_split=0.2, subset="both", seed=12345)
    
    def train(self):
        self.model.fit(self.train_data, validation_data=self.test_data, epochs=10)


geo_cnn = GeoCnn()
geo_cnn.load_dataset()
geo_cnn.setup_model()
geo_cnn.train()