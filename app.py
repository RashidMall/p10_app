from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64
import tensorflow as tf
from tensorflow import keras
from transformers import TFSegformerForSemanticSegmentation

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256
NUMBER_CLASSES = 8
IMAGE_PATH = './data/image'
MASK_PATH = './data/mask'

model = TFSegformerForSemanticSegmentation.from_pretrained("./segformer_mitb0.h5")
img_indexes = list(range(10))


def get_prediction(model, dataset, index=0):
    """
    """
    index = int(index)
    # Take a single sample from the dataset at the specified index
    sample = dataset.take(1).as_numpy_iterator().next()
    X = sample['pixel_values'][index:index + 1]
    Y = sample['labels'][index:index + 1]
    # Make a prediction using the model
    Y_PRED = model.predict(X)
    Y_PRED = tf.transpose(
        Y_PRED.logits,
        perm=[0, 2, 3, 1]
    )

    # Upsample the predicted mask
    Y_PRED = keras.layers.UpSampling2D(size=(4, 4))(Y_PRED)
    Y_PRED = tf.math.argmax(Y_PRED, axis=-1)

    # Transpose the input image for display
    X = tf.transpose(X, perm=[0, 2, 3, 1])

    for x, y, y_pred in zip(X, Y, Y_PRED):
        return image_mask_to_base64([x, y, y_pred], additional_title='Predicted Mask')


def image_mask_to_base64(display_list, additional_title=''):
    """
    Converts input images and their corresponding masks to a base64-encoded image.

    Parameters
    ----------
        display_list (list): List containing input images and masks as TensorFlow tensors or NumPy arrays.
        additional_title (str): Additional title for the plot.

    Returns
    -------
        str: Base64-encoded image.
    """
    titles = ['Original RGB Image', 'True Mask']

    if len(display_list) > len(titles):
        titles.append(additional_title)

    # Create subplots for visualization
    fig, axs = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    for i in range(len(display_list)):
        # Check if the element is a TensorFlow tensor and convert it to NumPy if needed
        display_image = display_list[i].numpy() if isinstance(display_list[i], tf.Tensor) else display_list[i]
        # Display original RGB image
        axs[i].imshow(display_image.astype(np.uint8), cmap='gist_rainbow')
        axs[i].set_title(titles[i])
        axs[i].grid(False)

    # Convert the Matplotlib figure to a base64-encoded image
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_data = base64.b64encode(img_buf.read()).decode()

    return img_data


def data_generator_HF(image_path, mask_path):
    """
    Generates a TensorFlow dataset for image segmentation tasks using the provided image and mask paths.

    Parameters
    ----------
        image_path (str): Path to the directory containing input images.
        mask_path (str): Path to the directory containing segmentation masks.

    Returns
    -------
        tf.data.Dataset: A TensorFlow dataset ready for training or testing.
    """
    batch_size = 16

    image_list = sorted([os.path.join(image_path, fname) for fname in os.listdir(image_path)])
    mask_list = sorted([os.path.join(mask_path, fname) for fname in os.listdir(mask_path)])

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data_HF, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset.prefetch(tf.data.AUTOTUNE)



def load_data_HF(image_list, mask_list):
    """
    Loads and preprocesses data for image segmentation tasks.

    Parameters
    ----------
        image_list (str): List of file paths for input images.
        mask_list (str): List of file paths for segmentation masks.

    Returns
    -------
        dict: A dictionary containing the preprocessed image and mask.
              Keys: "pixel_values" for the image and "labels" for the mask.
    """
    image = read_files(image_list)
    mask = read_files(mask_list, mask=True)

    # Transpose image dimensions for TensorFlow compatibility
    image = tf.transpose(image, (2, 0, 1))

    # Remove singleton dimensions from the mask
    mask = tf.squeeze(mask)

    return {
        "pixel_values": image,
        "labels": mask
    }


def read_files(file_path, mask=False):
    """
    Reads and preprocesses image or mask files.

    Args:
        file_path (str): File path for the image or mask.
        mask (bool): Whether the file is a segmentation mask.

    Returns:
        tf.Tensor: The preprocessed image or mask as a TensorFlow tensor.
    """
    image = tf.io.read_file(file_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)

        # Resize the mask using nearest-neighbor interpolation
        image = tf.image.resize(
            images=image,
            size=[IMAGE_HEIGHT, IMAGE_WIDTH],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        # Replace pixel values of 255 with 0
        image = tf.where(image == 255, np.dtype('uint8').type(0), image)
        image = tf.cast(image, tf.float32)
    else:
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(images=image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
        image = tf.cast(image, tf.float32)

    return image


def get_cats_data():
    data = [
        ('void', 10.65), 
        ('flat', 38.79), 
        ('construction', 22.08), 
        ('object', 1.85), 
        ('nature', 14.26), 
        ('sky', 3.56), 
        ('human', 1.24), 
        ('vehicle', 7.58)
    ]

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    return {'labels': labels, 'values': values}

def get_iou_data():
    data = [
        ('void', 0.5618), 
        ('flat', 0.9290), 
        ('construction', 0.8276), 
        ('object', 0.3747), 
        ('nature', 0.8767), 
        ('sky', 0.8737), 
        ('human', 0.5807), 
        ('vehicle', 0.8174)
    ]

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    return {'labels': labels, 'values': values}

def get_precision_data():
    data = [
        ('void', 0.6276), 
        ('flat', 0.9729), 
        ('construction', 0.9204), 
        ('object', 0.4637), 
        ('nature', 0.9461), 
        ('sky', 0.9717), 
        ('human', 0.7638), 
        ('vehicle', 0.9112)
    ]

    labels = [row[0] for row in data]
    values = [row[1] for row in data]

    return {'labels': labels, 'values': values}

# Data for chart plotting
cats_data = get_cats_data()
iou_data = get_iou_data()
precision_data = get_precision_data()

app = Flask(__name__, static_url_path='/')
app.secret_key = 'secret_key'

@app.route('/')
def index():
	print('index')

	return render_template('index.html', img_list=img_indexes, cats_data=cats_data, iou_data=iou_data, precision_data=precision_data)


@app.route('/predicted', methods=['POST', 'GET'])
def predicted():
    if request.method == 'POST':
        index = request.form['option']
        index = int(index)
        test_ds_hf = data_generator_HF(IMAGE_PATH, MASK_PATH)
        prediction = get_prediction(model, test_ds_hf, index)

        return render_template(
            'index.html', 
            img_list=img_indexes, 
            img_data=prediction, 
            img_index=index, 
            cats_data=cats_data,
            iou_data=iou_data,
            precision_data=precision_data
            )


if __name__ == "__main__":
    app.run()