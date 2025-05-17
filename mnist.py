import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')

x_train = x_train / 255
x_test = x_test / 255

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]

  )

model.fit(x_train,y_train, epochs = 10)
model.evaluate(x_test,y_test, verbose = 2)

model.save("mnist_model.h5")
print("Model saved as mnist_model.h5")

def draw_image():
    print("Draw your digit on the canvas below. Close the window when done.")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('off')
    canvas = np.ones((280, 280), dtype=np.uint8) * 255  # Blank white canvas
    im = ax.imshow(canvas, cmap='gray')

    def on_mouse_drag(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            # Draw a black circle
            rr, cc = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
            rr, cc = rr.flatten(), cc.flatten()
            mask = rr**2 + cc**2 <= 10**2
            rr, cc = rr[mask] + y, cc[mask] + x
            rr, cc = rr.clip(0, 279), cc.clip(0, 279)  # Keep within bounds
            canvas[rr, cc] = 0  # Set pixel values to black
            im.set_data(canvas)
            fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_drag)
    plt.show()
    return canvas

# Preprocess the drawn image
def preprocess_image(img):
    # Resize to 28x28
    img_resized = Image.fromarray(img).resize((28, 28))
    img_resized = ImageOps.grayscale(img_resized)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch dimension
    return img_array

# Main program for testing
def test_drawn_image():
    canvas = draw_image()
    preprocessed_img = preprocess_image(canvas)

    # Load the saved model
    loaded_model = tf.keras.models.load_model("mnist_model.h5")
    prediction = loaded_model.predict(preprocessed_img)
    predicted_digit = np.argmax(prediction)
    print(f"Predicted digit: {predicted_digit}")

    plt.imshow(canvas, cmap='gray')
    plt.title(f"Predicted digit: {predicted_digit}")
    plt.axis('off')
    plt.show()

# Uncomment the following line to test the model with a drawn digit
test_drawn_image()