import argparse
import tensorflow as tf
import json
import os
from datetime import datetime
from dataset import load_datasets
from seg_model.MyModel.SiameseInception_Keras import SiameseInception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from net_util import weight_binary_cross_entropy, dice_loss
from acc_util import F1_score#, Recall, Precision
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU, Recall, Precision, AUC
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow_addons.metrics import CohenKappa, F1Score

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='DSMSCN', help='model name')
parser.add_argument('--max_epoch', type=int, default=100, help='epoch to run[default: 100]')
parser.add_argument('--batch_size', type=int, default=8, help='batch size during training[default: 512]')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='initial learning rate[default: 1e-4]')
parser.add_argument('--model_save_path', default='saved_model/', help='model save path')
parser.add_argument('--checkpoint_path', default='checkpoints/', help='model checkpoint path')
parser.add_argument('--data_path', default='/app/spacenet7/csvs/sn7_baseline_train_df.csv', help='data path')
parser.add_argument('--model_path', default='/app/models/', help='path to the model dir')

# basic params
FLAGS = parser.parse_args()

BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
MODEL_SAVE_PATH = FLAGS.model_save_path
CHECKPOINT_PATH = FLAGS.checkpoint_path
DATA_PATH = FLAGS.data_path
MODEL_PATH = FLAGS.model_path
MODEL_NAME = FLAGS.model_name

dataset_train, dataset_val = load_datasets(DATA_PATH, batch_size=BATCH_SZ)
input_shape = [512, 512, 3]
siam_incep = SiameseInception()
model = siam_incep.get_model(input_shape)
print('Dataset spec')
print(dataset_train.element_spec)

precision = Precision()
recall = Recall()
#accuracy = BinaryAccuracy()
#f1_score = F1Score(num_classes=2, threshold=0.5)
f1_score = F1_score
kappa = CohenKappa(num_classes=2)
auc = AUC(num_thresholds=20)
iou = MeanIoU(num_classes=2)
# use LR?

model.compile(optimizer='adam',
                       loss=dice_loss, metrics=['accuracy', recall, precision, F1_score, iou])
#model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[accuracy, recall, precision])
model_checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor='val_loss')
early_stopping = EarlyStopping(patience=10)

train = True
# 1024 * 2 = 2048 upscaled image
# 2048 / 512 = 4 patches per side
# 4^2 = 16 patches per image
# 16 / 8 = 2 batches per image
if train:
    model_history = model.fit(dataset_train,
        validation_data=dataset_val.take(10),
        epochs=MAX_EPOCH,
        callbacks=[model_checkpoint_callback, early_stopping],
    )


now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
training_dir = os.path.join(MODEL_PATH, 'training_' + now)
os.mkdir(training_dir)
config_file = os.path.join(training_dir, 'config.json')
config = json.dumps({
    'model': MODEL_NAME,
    'batch_size': BATCH_SZ,
    'max_epoch': MAX_EPOCH,
    'lr': LEARNING_RATE
})
with open(config_file, 'w') as f:
    f.write(config)

history_dict = model_history.history if train else {}
history_file = os.path.join(training_dir, 'history.json')
json.dump(history_dict, open(history_file, 'w'))

saved_model_path = os.path.join(training_dir, MODEL_SAVE_PATH)
print('Saving model to', saved_model_path)
model.save(saved_model_path)
print('Converting to TFLite model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print('Saving TFLite model.tflite')
tflite_model_file = os.path.join(training_dir, 'model.tflite')
# Save the model.
with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)
