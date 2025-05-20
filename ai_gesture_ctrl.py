import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os
import time


IMG_SIZE = 64
NUM_CLASSES = 5
GESTURE_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "No Gesture"
}

def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    model = Sequential([
        # First layer
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        # Second layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        # Third layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.5),  # Add dropout to prevent overfitting
        Dense(NUM_CLASSES, activation='softmax')  # Output layer with 5 classes
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def preprocess_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    normalized = resized / 255.0

    preprocessed = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    return preprocessed

def collect_training_data():

    data_dir = "gesture_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        for i in range(NUM_CLASSES):
            os.makedirs(os.path.join(data_dir, str(i)), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nData Collection for Hand Gesture Recognition")
    print("--------------------------------------------")
    print("We'll collect images for each class:")
    for i in range(NUM_CLASSES - 1):
        print(f"Class {i}: {GESTURE_LABELS[i]}")
    print(f"Class {NUM_CLASSES - 1}: {GESTURE_LABELS[NUM_CLASSES - 1]} (no hand/default)")

    for gesture_class in range(NUM_CLASSES):
        input(f"\nGet ready to collect data for '{GESTURE_LABELS[gesture_class]}'. Press Enter when ready...")
        print(f"Collecting data for {GESTURE_LABELS[gesture_class]} in 3 seconds...")
        time.sleep(3)

        image_count = 0
        max_images = 500

        print(f"Show the '{GESTURE_LABELS[gesture_class]}' gesture to the webcam. Press 'q' to stop.")

        while image_count < max_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            display_frame = frame.copy()
            cv2.putText(display_frame, f"Class: {GESTURE_LABELS[gesture_class]} ({image_count}/{max_images})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Data Collection', display_frame)

            filename = os.path.join(data_dir, str(gesture_class), f"image_{image_count}.jpg")
            cv2.imwrite(filename, frame)
            image_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"Collected {image_count} images for '{GESTURE_LABELS[gesture_class]}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")


def load_training_data(data_dir="gesture_data"):
    images = []
    labels = []

    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found. Run collect_training_data() first.")
        return None, None

    for gesture_class in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(gesture_class))

        if not os.path.exists(class_dir):
            print(f"Warning: Directory for class {gesture_class} not found.")
            continue

        print(f"Loading data for class {gesture_class}: {GESTURE_LABELS[gesture_class]}")

        for filename in os.listdir(class_dir):
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            normalized = resized / 255.0

            images.append(normalized)
            labels.append(gesture_class)

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels)

    y_onehot = tf.keras.utils.to_categorical(y, NUM_CLASSES)

    print(f"Loaded {len(images)} images across {NUM_CLASSES} classes")

    return X, y_onehot


def train_model(epochs=15, batch_size=32, validation_split=0.2):

    X, y = load_training_data()

    if X is None or y is None:
        return None

    model = create_model()

    model.summary()

    print("\nTraining the model...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )

    model.save("hand_gesture_model.h5")
    print("Model saved as 'hand_gesture_model.h5'")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return model

def run_gesture_recognition():

    try:
        model = tf.keras.models.load_model("hand_gesture_model.h5")
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model. Please train the model first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\nHand Gesture Recognition Running...")
    print("Press 'q' to quit.")
    print("Recognized gestures will control the car:")
    for i in range(NUM_CLASSES):
        print(f"- {GESTURE_LABELS[i]}")

    confidence_threshold = 0.7

    prediction_history = []
    smoothing_window = 5

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        display_frame = frame.copy()

        preprocessed = preprocess_image(frame)

        prediction = model.predict(preprocessed, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        prediction_history.append(predicted_class)
        if len(prediction_history) > smoothing_window:
            prediction_history.pop(0)

        smoothed_class = max(set(prediction_history), key=prediction_history.count)

        status_text = f"Prediction: {GESTURE_LABELS[smoothed_class]}"
        confidence_text = f"Confidence: {confidence:.2f}"

        bar_width = int(200 * confidence)
        cv2.rectangle(display_frame, (10, 45), (10 + bar_width, 65), (0, 255, 0), -1)
        cv2.rectangle(display_frame, (10, 45), (210, 65), (0, 0, 0), 2)

        status_color = (0, 255, 0) if confidence > confidence_threshold else (0, 0, 255)

        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(display_frame, confidence_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        command_text = "Gesture: "
        if confidence > confidence_threshold:
            if smoothed_class == 0:
                command_text += "zero"
                cv2.putText(display_frame, "0", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif smoothed_class == 1:
                command_text += "one"
                cv2.putText(display_frame, "1", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif smoothed_class == 2:
                command_text += "two"
                cv2.putText(display_frame, "2", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif smoothed_class == 3:
                command_text += "three"
                cv2.putText(display_frame, "3", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                command_text += "No Gesture"
        else:
            command_text += "No Gesture (Low Confidence)"

        cv2.putText(display_frame, command_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('Hand Gesture Recognition', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to test the model on a few sample images
def test_model_on_samples(num_samples=5):
    try:
        model = tf.keras.models.load_model("hand_gesture_model.h5")
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model. Please train the model first.")
        return

    X, y = load_training_data()

    if X is None or y is None:
        return

    y_classes = np.argmax(y, axis=1)


    indices = np.random.choice(len(X), min(num_samples, len(X)), replace=False)

    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(indices):

        img = X[idx]
        true_label = y_classes[idx]

        img_for_pred = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(img_for_pred, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img.reshape(IMG_SIZE, IMG_SIZE), cmap='gray')

        title = f"True: {GESTURE_LABELS[true_label]}\n"
        title += f"Pred: {GESTURE_LABELS[predicted_class]}\n"
        title += f"Conf: {confidence:.2f}"

        color = 'green' if predicted_class == true_label else 'red'
        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()



def main():
    print("Welcome to the Hand Gesture Control System for AI Car!")
    print("------------------------------------------------------")
    print("Options:")
    print("1. Collect training data")
    print("2. Train the model")
    print("3. Test the model on sample images")
    print("4. Run real-time gesture recognition")
    print("5. Exit")

    while True:
        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            collect_training_data()
        elif choice == '2':
            train_model()
        elif choice == '3':
            test_model_on_samples()
        elif choice == '4':
            run_gesture_recognition()
        elif choice == '5':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()