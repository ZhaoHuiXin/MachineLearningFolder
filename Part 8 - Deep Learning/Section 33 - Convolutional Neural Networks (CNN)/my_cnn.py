# -*- coding: utf-8 -*-

# Prepare import package
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Prepare initializing CNN
classifier = Sequential()

# Adding the 1st convolultion layer to CNN
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))

# Adding the MaxPooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flattening
classifier.add(Flatten())

# Adding Full Connection (ANN)
classifier.add(Dense(units=128, activation='relu'))

# Adding Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
# Image Handler
## scaleing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                   zoom_range=0.2,  horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

## get more images
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

## fitting the classifier , and testing the test set
classifier.fit_generator(generator=training_set,
                         steps_per_epoch=250,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=62.5) 