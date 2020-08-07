from typing import Callable

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.utils import class_weight

def one_hot(label, num_classes):
    #label = rgb2gray(label)
    if np.ndim(label) == 3:
        
        label = np.squeeze(label, axis=-1)
    assert np.ndim(label) == 2

    heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))
    for i in range(num_classes):
        heat_map[:, :, i] = np.equal(label, i).astype('float32')
    return heat_map

def cal_chunk_number(total_size, batch_size):
    #print('##################################')
    #print(total_size)
    #print(batch_size)
    if total_size % batch_size == 0:
        return total_size // batch_size
    else:
        return (total_size // batch_size) + 1


class DataGenerator(Sequence):
    def __init__(self, data, *, batch_size: int, n_classes: int,
                 preprocess_image: Callable, shuffle: bool=False):
        """

        Args:
            data: input data
            batch_size(int): batch size
            n_classes(int): number of classes
            shuffle(bool):  Whether or not to shuffle the data after each epoch.
            preprocess_image: a function to preprocess a image file encoding a batch of images
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_classes = n_classes
        self._get_chunks()
        self.preprocess_image = preprocess_image

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        rows = self.chunks[index]
        batch_images, batch_labels = self.process_instances(rows)
        #print(np.unique(batch_images))
        #print(np.unique(batch_labels))
        
        
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)
            self._get_chunks()

    def process_instances(self, rows):
        batch_images = []
        batch_labels = []
        for row in rows:
            file_path, label_path = row[0], row[1]
            #x = self.preprocess_image(file_path)
            x, y = self.preprocess_image(file_path, label_path)
            #y = one_hot(y, number_class=4)
            batch_images.append(x)
            batch_labels.append(y)
        return np.concatenate(batch_images, axis=0), np.concatenate(batch_labels, axis=0)

    def get_epoch_num(self):
        return cal_chunk_number(len(self.data), self.batch_size)

    def _get_chunks(self):
        self.chunks = np.array_split(self.data, cal_chunk_number(len(self.data), self.batch_size))

    def class_weights(self):
        labels = []
        for row in self.data:
            _, label = row[0], row[1]
            labels.append(label)
        class_weight_list = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
        cw = dict(zip(np.unique(labels), class_weight_list))
        return cw
