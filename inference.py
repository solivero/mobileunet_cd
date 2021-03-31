import numpy as np
import tensorflow as tf
import argparse
import os
import time
from dataset import load_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/app/spacenet7/csvs/sn7_baseline_train_df.csv', help='data path')
parser.add_argument('--saved_model', default='saved_model', help='saved model path')
parser.add_argument('--tflite_model', default='model.tflite', help='tflite model path')

args = parser.parse_args()

def postprocess(pred):
    if pred.ndim == 3:
        pred = np.expand_dims(pred, axis=0)
    print(pred.shape)
    pred[pred > 0.5] = 255
    pred[pred <= 0.5] = 0
    predictions_dir = './predictions'
    if not os.path.exists(predictions_dir):
        os.mkdir(predictions_dir)
    for i, img in enumerate(pred):
        cast_img = tf.image.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
        cast_img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
        png_img = tf.io.encode_png(cast_img)
        mask_path = os.path.join(predictions_dir, f'mask-{i}.png')
        tf.io.write_file(
            mask_path, png_img, name=None
        )

def keras_inference():
    _, dataset_val = load_datasets(args.data_path)
    model = tf.keras.models.load_model(args.saved_model)
    pred = model.predict(dataset_val.take(8*2))

def lite_inference():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    tic = time.perf_counter()
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    toc = time.perf_counter()
    print(output_data.shape)
    print(output_data)
    print(f'Ran inference in {toc - tic:0.4f} seconds')
    postprocess(output_data)

if __name__ == '__main__':
    #keras_inference()
    lite_inference()