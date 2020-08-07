
from tensorflow.keras import datasets, layers, models, Model
from tensorflow.keras.applications import inception_v3, densenet, inception_resnet_v2, nasnet, resnet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2DTranspose, Conv2D
import numpy as np

label = np.ones((2,3))
num_classes = 4
heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))

print(np.shape(heat_map))

'''
def conv_relu( x, filters, kernel_size=1):
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.ReLU()(x)
    return x

n_classes=2
input_shape=(224, 224, 3)
base_model = resnet.ResNet50(weights='imagenet', include_top=True, input_shape=input_shape)
base_model.summary()


base_model = resnet.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)
x = model.output
x = conv_relu(x, 2048, 5)
x = Dropout(rate=0.5)(x)
x = conv_relu(x, 2048, 1)
x = Dropout(rate=0.5)(x)

x = Conv2D(n_classes, 1, kernel_initializer='he_normal')(x)
seg = Conv2DTranspose(n_classes, 64, strides=32, padding='same', kernel_initializer='he_normal')(x)
final_model = Model(inputs=model.input, outputs=seg)

final_model.summary()
'''

'''
model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
final_model = Model(inputs=model.input, outputs=predictions)
return final_model
'''


