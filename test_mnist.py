import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tensorflow as tf
from PIL import Image, ImageOps
import argparse
import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg backend for better interactive support


def draw_image():
    """
    Create an interactive canvas to draw a digit with immediate visual feedback.
    Returns the drawn image as a numpy array.
    """
    print("Draw your digit on the canvas below. Close the window when done.")

    # Set up the canvas
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    canvas = np.ones((280, 280), dtype=np.uint8) * 255  # Blank white canvas
    img_display = ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)

    # Set up drawing variables
    drawing = False
    last_x, last_y = None, None

    # Handle mouse events for drawing
    def on_mouse_press(event):
        nonlocal drawing, last_x, last_y
        if event.inaxes == ax:  # Check if mouse is inside axes
            drawing = True
            last_x, last_y = int(event.xdata), int(event.ydata)

            # Draw initial point
            draw_point(last_x, last_y)
            img_display.set_data(canvas)
            fig.canvas.draw_idle()  # Update the canvas

    def on_mouse_release(event):
        nonlocal drawing, last_x, last_y
        drawing = False
        last_x, last_y = None, None

    def on_mouse_motion(event):
        nonlocal drawing, last_x, last_y
        if drawing and event.inaxes == ax:  # Check if mouse is inside axes and drawing
            x, y = int(event.xdata), int(event.ydata)

            # Draw line between last point and current point
            if last_x is not None and last_y is not None:
                draw_line(last_x, last_y, x, y)

            # Update last position
            last_x, last_y = x, y

            # Update display
            img_display.set_data(canvas)
            fig.canvas.draw_idle()  # Force canvas update

    def draw_point(x, y, radius=8):
        """Draw a filled circle at (x,y) with given radius"""
        y_grid, x_grid = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x_grid ** 2 + y_grid ** 2 <= radius ** 2

        # Get indices within canvas bounds
        y_indices, x_indices = np.where(mask)
        y_indices = y_indices - radius + y
        x_indices = x_indices - radius + x

        # Keep only indices that are within canvas bounds
        valid = (y_indices >= 0) & (y_indices < canvas.shape[0]) & \
                (x_indices >= 0) & (x_indices < canvas.shape[1])

        y_indices, x_indices = y_indices[valid], x_indices[valid]
        canvas[y_indices, x_indices] = 0  # Set to black

    def draw_line(x1, y1, x2, y2, thickness=8):
        """Draw a line from (x1,y1) to (x2,y2) with given thickness"""
        # Calculate points along the line
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length == 0:
            draw_point(x1, y1, thickness)
            return

        x_points = np.linspace(x1, x2, length * 3)
        y_points = np.linspace(y1, y2, length * 3)

        # Draw points along the line
        for x, y in zip(x_points, y_points):
            draw_point(int(x), int(y), thickness)

    def clear_canvas(event):
        nonlocal canvas
        canvas = np.ones((280, 280), dtype=np.uint8) * 255
        img_display.set_data(canvas)
        fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

    # Add clear button
    ax_clear = plt.axes([0.7, 0.02, 0.2, 0.05])
    btn_clear = Button(ax_clear, 'Clear Drawing')
    btn_clear.on_clicked(clear_canvas)

    # Add title with instructions
    plt.suptitle("Draw a digit (0-9) here\nClose window when finished", fontsize=16)

    plt.tight_layout()
    plt.show()

    return canvas


def preprocess_image(img):
    """
    Preprocess the drawn image for the MNIST model:
    - Resize to 28x28
    - Invert colors (MNIST digits are white on black)
    - Normalize to [0,1]
    - Reshape for model input
    """
    # Center the digit by finding its bounding box
    rows = np.any(img < 255, axis=1)
    cols = np.any(img < 255, axis=0)

    if np.any(rows) and np.any(cols):
        # Find the bounding box
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add padding
        padding = 20
        rmin = max(0, rmin - padding)
        rmax = min(img.shape[0] - 1, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(img.shape[1] - 1, cmax + padding)

        # Crop to the bounding box
        img_cropped = img[rmin:rmax + 1, cmin:cmax + 1]

        # Create a square image with padding
        size = max(img_cropped.shape)
        img_square = np.ones((size, size), dtype=np.uint8) * 255

        # Center the cropped image in the square
        offset_r = (size - img_cropped.shape[0]) // 2
        offset_c = (size - img_cropped.shape[1]) // 2
        img_square[offset_r:offset_r + img_cropped.shape[0],
        offset_c:offset_c + img_cropped.shape[1]] = img_cropped

        # Use the centered square image
        img = img_square

    # Save the original processed image before resizing
    plt.imsave('drawn_digit_original.png', img, cmap='gray')

    # Resize to 28x28
    img_pil = Image.fromarray(img).resize((28, 28), Image.LANCZOS)

    # Keep as grayscale
    img_pil = ImageOps.grayscale(img_pil)

    # Convert to array and normalize
    img_array = np.array(img_pil) / 255.0

    # MNIST expects the opposite polarity, so invert if the background is white
    if np.mean(img_array) > 0.5:
        img_array = 1.0 - img_array

    # Save the processed image for inspection
    plt.imsave('drawn_digit_processed.png', img_array, cmap='gray')

    # Reshape for model input (batch_size, height, width, channels)
    img_for_model = img_array.reshape(1, 28, 28, 1)

    # Also keep original for display
    img_for_display = img_array

    return img_for_model, img_for_display


def load_model(model_path):
    """
    Load the MNIST model from the specified path
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_digit(model, img_array):
    """
    Make a prediction using the model and return probabilities
    """
    if model is None:
        return None, None

    # Get prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(predictions[0])
    probabilities = predictions[0]

    return predicted_digit, probabilities


def display_prediction(img, predicted_digit, probabilities):
    """
    Display the preprocessed image and prediction results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display the preprocessed image
    ax1.imshow(img, cmap='gray')
    ax1.set_title("Preprocessed Image")
    ax1.axis('off')

    # Display prediction probabilities as a bar chart
    digits = np.arange(10)
    ax2.bar(digits, probabilities)
    ax2.set_xticks(digits)
    ax2.set_xlabel("Digit")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"Prediction: {predicted_digit}")
    ax2.grid(axis='y', alpha=0.3)

    # Highlight the predicted digit
    ax2.get_children()[predicted_digit].set_color('red')

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition')
    parser.add_argument('--model', type=str, default='mnist_model.h5',
                        help='Path to the MNIST model .h5 file')
    parser.add_argument('--draw', action='store_true',
                        help='Draw a digit to test the model')
    parser.add_argument('--image', type=str,
                        help='Path to an image file containing a digit to test')

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Process input based on arguments
    if args.draw or (not args.image):  # Default to drawing if no specific input
        # Draw a digit
        print("Starting drawing interface... Please wait while it initializes.")
        canvas = draw_image()
        print("Processing your drawing...")
        img_for_model, img_for_display = preprocess_image(canvas)
    elif args.image:
        # Load image from file
        try:
            print(f"Loading image from {args.image}...")
            img = np.array(Image.open(args.image).convert('L'))
            img_for_model, img_for_display = preprocess_image(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return

    # Make prediction
    print("Making prediction...")
    predicted_digit, probabilities = predict_digit(model, img_for_model)

    if predicted_digit is not None:
        print(f"Predicted digit: {predicted_digit}")
        print(f"Confidence: {probabilities[predicted_digit]:.4f}")
        display_prediction(img_for_display, predicted_digit, probabilities)


if __name__ == "__main__":
    main()