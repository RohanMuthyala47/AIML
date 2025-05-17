import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import numpy as np
from tensorflow.keras.preprocessing import image

Training_Data_Set = "C:/Users/Rohan/OneDrive/Desktop/Main_Dataset/Training"
Testing_Data_Set = "C:/Users/Rohan/OneDrive/Desktop/Main_Dataset/Testing"

processing_parameters = ImageDataGenerator(
    rescale = 1.0/255,
    rotation_range = 10,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_processing_parameters = ImageDataGenerator(rescale = 1.0/255)

new_training_images = processing_parameters.flow_from_directory(
    Training_Data_Set,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

new_testing_images = test_processing_parameters.flow_from_directory(
    Testing_Data_Set,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(15, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    new_training_images,
    steps_per_epoch=new_training_images.samples // new_training_images.batch_size,
    epochs=30,
    validation_data=new_testing_images,
    validation_steps=new_testing_images.samples // new_testing_images.batch_size,
    verbose=2
)

training_Loss, training_accuracy = model.evaluate(new_testing_images)
print(f"Validation Accuracy: {training_accuracy * 100:.2f}%")