import tensorflow as tf

import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

root = "/app"
dataset_path = os.path.join(root, "spacenet7")
training_data = "train/"
val_data = "train/"
IMG_SIZE = 1024 # Image size expected on file
UPSCALE = 2 # To avoid tiny areas of 2x2 pixels
PATCH_SIZE = 512 # After upscaling
SEED = 42

def parse_image_pair(csv_batch) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    img1_path = csv_batch['image1'][0]
    image1 = tf.io.read_file(img1_path)
    image1 = tf.image.decode_png(image1)
    image1 = tf.image.convert_image_dtype(image1, tf.float32)[:, :, :3]

    img2_path = csv_batch['image2'][0]
    image2 = tf.io.read_file(img2_path)
    image2 = tf.image.decode_png(image2)
    image2 = tf.image.convert_image_dtype(image2, tf.float32)[:, :, :3]

    #cm_name = tf.strings.regex_replace(mask_path, r'20\d{2}_\d{2}', double_date)
    cm_name = csv_batch['label'][0]

    #cm_name = mask_path

    mask = tf.io.read_file(cm_name)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)[:, :, :1]
    #mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
    #filler_row = tf.zeros((1, 1024, 1), tf.uint8)
    #mask = tf.concat([mask, filler_row], axis=0)

    # Note that we have to convert the new value (0)

    merged_image = tf.concat([image1, image2], axis=2)
    #filler_row = tf.zeros((1, 1024, 6), tf.uint8)
    #merged_image = tf.concat([merged_image, filler_row], axis=0)

    #return {'image': merged_image, 'segmentation_mask': mask}
    return image1, image2, mask

@tf.function
def make_patches(image1: tf.Tensor, image2: tf.Tensor, mask: tf.Tensor):
    n_patches = ((IMG_SIZE*UPSCALE) // PATCH_SIZE)**2
    image1_patches = tf.image.extract_patches(images=tf.expand_dims(image1, 0),
                        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')[0]
    print(image1_patches.shape)
    image1_patch_batch = tf.reshape(image1_patches, (n_patches, PATCH_SIZE, PATCH_SIZE, 3))
    image2_patches = tf.image.extract_patches(images=tf.expand_dims(image2, 0),
                        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')[0]
    print(image2_patches.shape)
    image2_patch_batch = tf.reshape(image2_patches, (n_patches, PATCH_SIZE, PATCH_SIZE, 3))
    mask_patches = tf.image.extract_patches(images=tf.expand_dims(mask, 0),
                        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')[0]
    mask_patch_batch = tf.reshape(mask_patches, (n_patches, PATCH_SIZE, PATCH_SIZE, 1))
    return image1_patch_batch, image2_patch_batch, mask_patch_batch



#val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.tif", seed=SEED)
#val_dataset = val_dataset.map(parse_image)

@tf.function
def normalize(input_image1: tf.Tensor, input_image2: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image1 = tf.cast(input_image1, tf.float32) / 255.0
    input_image2 = tf.cast(input_image2, tf.float32) / 255.0
    return input_image1, input_image2, input_mask

@tf.function
def upscale_images(image1: tf.Tensor, image2: tf.Tensor, mask: tf.Tensor) -> tuple:
    upscaled_size = IMG_SIZE*UPSCALE
    # use nearest neightbor?
    input_image1 = tf.image.resize(image1, (upscaled_size, upscaled_size))
    input_image2 = tf.image.resize(image2, (upscaled_size, upscaled_size))
    input_mask = tf.image.resize(mask, (upscaled_size, upscaled_size))
    return input_image1, input_image2, input_mask

@tf.function
def load_image_train(image1: tf.Tensor, image2: tf.Tensor, mask: tf.Tensor) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """

    if tf.random.uniform(()) > 1:
        image1 = tf.image.flip_left_right(image1)
        image2 = tf.image.flip_left_right(image2)
        mask = tf.image.flip_left_right(mask)

    #input_image1, input_image2, input_mask = normalize(image1, image2, mask)

    return {'input_1': image1, 'input_2': image2}, mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (PATCH_SIZE, PATCH_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (PATCH_SIZE, PATCH_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

BUFFER_SIZE = 100

def load_image_dataset(csv_dataset):
    return csv_dataset \
        .map(parse_image_pair) \
        .map(upscale_images) \
        .flat_map(lambda image1, image2, mask: tf.data.Dataset.from_tensor_slices(make_patches(image1, image2, mask))) \
        .map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

def load_csv_dataset(csv_path):
    return tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=1, # Actual batching in later stages
        num_epochs=1,
        ignore_errors=True)
        # Shuffle train_csv_ds first to have diverse val set?

def load_datasets(csv_path, batch_size=8, val_size=256, buffer_size=100):
    csv_dataset = load_csv_dataset(csv_path)
    train_csv = csv_dataset.skip(val_size)
    dataset_train = load_image_dataset(train_csv) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)
        #.shuffle(buffer_size=buffer_size, seed=SEED) \
    val_csv = csv_dataset.take(val_size)
    dataset_val = load_image_dataset(val_csv) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)
    return dataset_train, dataset_val

if __name__ == '__main__':
    train_ds, val_ds = load_datasets('/app/spacenet7/csvs/sn7_baseline_train_direct_class.csv')

    predictions_dir = './training_samples'
    def save_img(img, name):
        cast_img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
        png_img = tf.io.encode_png(cast_img)
        mask_path = os.path.join(predictions_dir, name)
        tf.io.write_file(
            mask_path, png_img, name=None
        )
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
    for b, (img_batch, mask_batch) in enumerate(train_ds.take(4)):
        for i, img1, img2, mask in zip(range(len(img_batch)), img_batch['input_1'], img_batch['input_2'], mask_batch):
            #cast_img = tf.image.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
            save_img(img1, f'b{b}-i{i}-img1.png')
            save_img(img2, f'b{b}-i{i}-img2.png')
            save_img(mask, f'b{b}-i{i}-mask.png')
