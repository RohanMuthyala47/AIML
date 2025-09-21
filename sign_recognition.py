import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Set dataset and test image directories
dataset_dir = "C:/Users/Rohan/Downloads/Dataset"
test_image_dir = "C:/Users/Rohan/Downloads/TestImages"
image_size = (100, 100)
num_classes = 5
model_save_path = "digit_classifier_model.h5"

# Load training images
def load_images(dataset_dir):
    images = []
    labels = []
    label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

    for label_name, label in label_map.items():
        folder_path = os.path.join(dataset_dir, label_name)
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = image.load_img(img_path, target_size=image_size, color_mode='rgb')
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(label)

    images = np.array(images) / 255.0
    labels = np.array(labels)
    return images, labels

# Manual train/test split using numpy
def custom_train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]

# Load and split data
images, labels = load_images(dataset_dir)
X_train, X_test, y_train, y_test = custom_train_test_split(images, labels)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate and save
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training history
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# 🔍 Function to predict test images from folder
def predict_images_from_folder(folder_path, model_path, image_size=(100, 100), class_labels=['0', '1', '2', '3', '4']):
    model = tf.keras.models.load_model(model_path)
    print("Model loaded for prediction.")

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(folder_path, file_name)
        img = image.load_img(img_path, target_size=image_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        plt.imshow(img)
        plt.title(f"Predicted: {class_labels[predicted_class]}")
        plt.axis('off')
        plt.show()


# 🔍 Run prediction on new test images
predict_images_from_folder(test_image_dir, model_save_path)
