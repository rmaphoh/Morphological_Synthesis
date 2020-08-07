import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from matplotlib import pyplot as plt
import datetime
import os
import cv2
import argparse
import multiprocessing
from pathlib import Path
import pandas as pd
import Ophthal_model
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
import numpy as np
import pathlib
import random
from Data_Generator import DataGenerator
from tensorflow.keras.preprocessing import image
# CUDNN handle error - allow GPU growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras_preprocessing import image as keras_image

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


parser=argparse.ArgumentParser()
parser.add_argument(
    '--model_name',
    type=str,
    required=True
    )

parser.add_argument(
    '--train_directory',
    type=str,
    required=True
    )

parser.add_argument(
    '--test_directory',
    type=str,
    required=True
    )

parser.add_argument(
    '--batch_size',
    type=int,
    required=True
    )

parser.add_argument(
    '--checkpoint_path',
    type=str,
    required=True
    )

FLAGS,_= parser.parse_known_args()

batch_size = FLAGS.batch_size

data_directory_file = FLAGS.train_directory
train_directory_file = pathlib.Path(data_directory_file)

data_directory_file = FLAGS.test_directory
test_directory_file = pathlib.Path(data_directory_file)

log_dir = './logs'
checkpoint_path = FLAGS.checkpoint_path
#checkpoint_path = pathlib.Path(checkpoint_path)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def one_hot(label, num_classes):
    if np.ndim(label) == 3:
        label = np.squeeze(label, axis=-1)
    assert np.ndim(label) == 2

    heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))
    for i in range(num_classes):
        #heat_map[:, :, i] = np.equal(label, int(i*127.5)).astype('float32')
        heat_map[:, :, i] = np.equal(label, i).astype('float32')
    return heat_map


def random_horizontal_flip(image, label):
    if True:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label


def random_vertical_flip(image, label):
    if True:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
    return image, label


def random_rotation(image, label, rotation_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if rotation_range > 0.:
        theta = np.random.uniform(-rotation_range, rotation_range)
        # rotate it!
        image = keras_image.apply_affine_transform(image, theta=theta, fill_mode='nearest')
        label = keras_image.apply_affine_transform(label, theta=theta, fill_mode='nearest')
    return image, label


def preprocess_image(file_path, label_path):
    #img = crop2square(image.load_img(image_path)).resize((512, 512))
    image_1 = image.load_img(file_path).resize((256, 256))
    image_1 = image.img_to_array(image_1)
    label = image.load_img(label_path,color_mode='grayscale').resize((256, 256))
    label = image.img_to_array(label)
    
    if True:
        # random vertical flip
        if np.random.randint(2):
            image_1, label = random_vertical_flip(image_1, label)
        # random horizontal flip
        if np.random.randint(2):
            image_1, label = random_horizontal_flip(image_1, label)
        # random brightness
        #if np.random.randint(2):
        #    image_1, label = random_brightness(image_1, label)
        # random rotation
        #if np.random.randint(2):
        #    image_1, label = random_rotation(image_1, label, rotation_range=60)
    
    image_1 = np.expand_dims(image_1, axis=0)
    
    #print(np.unique(x))
    image_1 /= 127.5
    image_1 -= 1.
    #label /= 127.5
    #label -= 1.
    #print(np.shape(label))
    #label = one_hot(label, num_classes=4)
    label = one_hot(label, num_classes=3)
    label = np.expand_dims(label, axis=0)

    return image_1, label


'''

def preprocess_image(file_path, label_path):
    #img = crop2square(image.load_img(image_path)).resize((512, 512))
    image_1 = image.load_img(file_path).resize((256, 256))
    image_1 = image.img_to_array(image_1)
    label = image.load_img(label_path).resize((256, 256))
    label = image.img_to_array(label)
    
    if True:
        # random vertical flip
        if np.random.randint(2):
            image_1, label = random_vertical_flip(image_1, label)
        # random horizontal flip
        if np.random.randint(2):
            image_1, label = random_horizontal_flip(image_1, label)
        # random brightness
        #if np.random.randint(2):
        #    image_1, label = random_brightness(image_1, label)
        # random rotation
        #if np.random.randint(2):
        #    image_1, label = random_rotation(image_1, label, rotation_range=60)
    
    image_1 = np.expand_dims(image_1, axis=0)
    label = np.expand_dims(label, axis=0)
    
    #print(np.unique(x))
    image_1 /= 127.5
    image_1 -= 1.
    label /= 127.5
    label -= 1.
    #print(np.shape(label))
    #label = one_hot(label, num_classes=4)
    #label = one_hot(label, num_classes=3)
    #label = np.expand_dims(label, axis=0)

    return image_1, label
'''

# ds_trian could be "image_batch, label_batch = next(iter(ds_trian))



def train(model, train_data, valid_data, best_model, batch_size=16, n_classes=2):
    #early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=K.epsilon(), patience=5, verbose=1)
    best_model_cp = callbacks.ModelCheckpoint(best_model, save_best_only=True, save_weights_only=False, monitor='val_accuracy', verbose=1)
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    #optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=0.0, decay=0.0)

    if n_classes == 2:
        model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    cpu_count = multiprocessing.cpu_count()
    workers = max(int(cpu_count / 3), 1)

    train_generator = DataGenerator(train_data, preprocess_image=preprocess_image, batch_size=batch_size, n_classes=n_classes, shuffle=True)
    valid_generator = DataGenerator(valid_data, preprocess_image=preprocess_image,
                                    batch_size=batch_size, n_classes=n_classes, shuffle=False)

    train_chunck_number = train_generator.get_epoch_num()
    print(train_chunck_number)
    
    
    history = model.fit(
                        train_generator,
                        use_multiprocessing=True,
                        workers=workers,
                        steps_per_epoch=train_chunck_number,
                        #callbacks=[early_stop, best_model_cp],
                        callbacks=[ best_model_cp, tensorboard_callback],
                        epochs=400,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.get_epoch_num(),
                        verbose=1)

    return history


def visulization(history):
    dfhistory = pd.DataFrame(history.history)
    dfhistory.index = range(1,len(dfhistory) +1)
    dfhistory.index.name = 'epoch'
    print(dfhistory)

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) +1 )
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro--')
    plt.title('training and validation '+ metric)
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.legend(['train_' +metric, "val_"+metric])
    plt.show()


def prep_instances(train_file, val_size=0.8, shuffle=True, parent='.'):
    """
    Read the dataset.

    Args:
        train_file(str): the file path of training dataset
        val_size(float): should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
            validation split.
        shuffle(bool): Whether or not to shuffle the data before splitting.
        parent(str): data directory
    Returns:
        list: List containing train-validation split of inputs.
    """
    df = pd.read_csv(train_file)
    rows = df.values.tolist()

    for i in range(len(rows)):
        rows[i][0] = os.path.join(parent, rows[i][0])

    if shuffle:
        np.random.shuffle(rows)
    split_size = int(len(rows) * val_size)
    return rows[:split_size], rows[split_size:]



if __name__ == '__main__':

    n_classes = 3

    model = Ophthal_model.Model_select(model_name=FLAGS.model_name, n_classes=n_classes, input_shape=(256, 256, 3))

    model.summary()
    #train_data_list, valid_data_list = direction_distribute(data_directory_file, val_size=0.8, shuffle=True)

    train_data, valid_data = prep_instances(train_directory_file, parent='.')

    #print('train_data shape ',np.shape(train_data))
    #print('valid_data shape ',np.shape(valid_data))


    history_record = train(model, train_data, valid_data, checkpoint_path, batch_size=batch_size, n_classes=n_classes)

    visulization(history_record)

    plot_metric(history_record,'loss')

    plot_metric(history_record,'accuracy')


