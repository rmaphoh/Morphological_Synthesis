import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import inception_v3, densenet, inception_resnet_v2, nasnet, resnet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Activation
import numpy as np
from tensorflow.keras.layers import Concatenate, Dense, Multiply, Lambda, Input, Add, LeakyReLU, Flatten, BatchNormalization
from tensorflow.keras.models import Model


def Model_select(model_name, n_classes=1, input_shape=(256, 256, 3)):
    
    input_shapes = input_shape
    n_class = n_classes

    if model_name =='inveptionV3':
        return InceptionV3Model(n_classes=n_class, input_shape=input_shapes)
    if model_name =='InceptionV3Model_fl':
        return InceptionV3Model_fl(n_classes=n_class, input_shape=input_shapes)
    if model_name =='DensenetModel':
        return DensenetModel(n_classes=n_class, input_shape=input_shapes)
    if model_name =='DensenetModel_fl':
        return DensenetModel_fl(n_classes=n_class, input_shape=input_shapes)
    if model_name =='NasnetModel':
        return NasnetModel(n_classes=n_class, input_shape=input_shapes)
    if model_name =='InceptionResNetV2Model':
        return InceptionResNetV2Model(n_classes=n_class, input_shape=input_shapes)
    if model_name =='InceptionResNetV2Model_fl':
        return InceptionResNetV2Model_fl(n_classes=n_class, input_shape=input_shapes)
    if model_name =='Inceptionv3_cam':
        return Inceptionv3_cam(n_classes=n_class, input_shape=input_shapes)
    if model_name == 'Resnet':
        return ResNet50(n_classes=n_class, input_shape=input_shapes)
    if model_name == 'Unet':
        return Unet(n_classes=n_class, input_shape=input_shapes)
    if model_name == 'deep_Unet':
        return deep_Unet(n_classes=n_class, input_shape=input_shapes)


def conv_relu( x, filters, kernel_size=1):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.ReLU()(x)
    return x


def ResNet50(n_classes=1, input_shape=(256, 256, 3)):
    base_model = resnet.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_out').output)
    x = model.output
    #x = conv_relu(x, 2048, 5)
    #x = Dropout(rate=0.5)(x)
    x = conv_relu(x, 2048, 1)
    x = Dropout(rate=0.5)(x)

    #x = Conv2D(n_classes, 1)(x)
    #seg = Conv2DTranspose(n_classes, 64, strides=32, activation='softmax', padding='same')(x)
    ###############################
    conv5 = conv_relu(x, 1024,  3)
    conv5 = UpSampling2D(size=(2, 2))(conv5)
    conv5 = conv_relu(conv5, 512,  3)
    conv5 = UpSampling2D(size=(2, 2))(conv5)
    conv5 = conv_relu(conv5, 256,  3)
    conv5 = UpSampling2D(size=(2, 2))(conv5)
    conv5 = conv_relu(conv5, 128,  3)
    conv5 = UpSampling2D(size=(2, 2))(conv5)
    conv5 = conv_relu(conv5, 64,  3)
    conv5 = UpSampling2D(size=(2, 2))(conv5)
    #conv5 = layers.Conv2D(filters=4, kernel_size=(3,3), padding='same')(conv5)
    conv5 = layers.Conv2D(filters=3, kernel_size=(3,3), padding='same')(conv5)
    seg = layers.Softmax()(conv5)

    final_model = Model(inputs=model.input, outputs=seg)
    return final_model


def squeeze_idt(idt):
    n, h, w, c = idt.get_shape().as_list()                                                    

    idt = tf.reshape(tensor=idt, shape=[tf.shape(idt)[0], h, w, c // 2, 2])     # maybe you need to use tf.shape() function, its returns tensor variable            

    return tf.reduce_sum(input_tensor=idt, axis=4) 


def Unet(n_classes=3, input_shape=(256,256,3)):
    """
    generate network based on unet
    """    
        
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    n_filters=32

    img_ch=3 # image channels
    out_ch=n_classes # output channel
    img_size = input_shape
    img_height, img_width = img_size[0], img_size[1]
    padding='same'
    
    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    
#upsampling1    
#################################################    
    conv5 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
###############################
    conv5 = UpSampling2D(size=(s, s))(conv5)
################################################################
    conv5 = Lambda(squeeze_idt, name='squeeze1_1')(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv4 = Lambda(squeeze_idt, name='squeeze1_2')(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    up12 = Concatenate(axis=3)([conv5, conv4])
    
#up122 is the attention module -20190423
    up122 = GlobalAveragePooling2D()(up12)
    up122 = Activation('sigmoid')(Multiply()([up12,up122]))
#    up12 = up12*up122
##################################################
###############################################    
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
###################################
    conv6 = Activation('relu')(conv6)
    conv6 = Add()([conv6, up122])

    
#upsampling2      
###########################################    
    conv6 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
##################################
    conv6 = UpSampling2D(size=(s, s))(conv6)
############################################################################
    conv6 = Lambda(squeeze_idt, name='squeeze2_1')(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv3 = Lambda(squeeze_idt, name='squeeze2_2')(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv3])
    up22 = Concatenate(axis=3)([conv6, conv3])
#up122 is the attention module -20190423
    up222 = GlobalAveragePooling2D()(up22)
    up222 = Activation('sigmoid')(Multiply()([up22,up222]))
#    up22 = up22*up222
###################################################################

    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
##############################################

    conv7 = Activation('relu')(conv7) 
    conv7 = Add()([conv7, up222])

    
#upsampling3      
#######################################################################    
    conv7 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)   
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
#####################################################################    
    conv7 = UpSampling2D(size=(s, s))(conv7)
#############################################################
    conv7 = Lambda(squeeze_idt, name='squeeze3_1')(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv2 = Lambda(squeeze_idt, name='squeeze3_2')(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv3])
    up32 = Concatenate(axis=3)([conv7, conv2])   
#up122 is the attention module -20190423
    up322 = GlobalAveragePooling2D()(up32)
    up322 = Activation('sigmoid')(Multiply()([up32,up322]))
#    up32 = up32*up322
###########################################################################    

    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
##########################################################################

    conv8 = Activation('relu')(conv8)
    conv8 = Add()([conv8, up322])

    
    
#upsampling4  
##############################################################
    conv8 = Conv2D(n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
############################################################################    
    conv8 = UpSampling2D(size=(s, s))(conv8)
    
    conv8 = Lambda(squeeze_idt, name='squeeze4_1')(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv1 = Lambda(squeeze_idt, name='squeeze4_2')(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv3])
    up42 = Concatenate(axis=3)([conv8, conv1])
#up122 is the attention module -20190423
    up422 = GlobalAveragePooling2D()(up42)
    up422 = Activation('sigmoid')(Multiply()([up42,up422]))
#    up42 = up42*up422
################################################################################    
    
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)    
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    

    conv9 = Activation('relu')(conv9)
    conv9 = Add()([conv9, up422])
    
    # this is for alignment in the value of input [-1, 1]
    #outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='softmax')(conv9)

    g = Model(inputs, outputs)
    
#    g = multi_gpu_model(g, gpus=2 )

    return g


'''

def Unet(n_classes=3, input_shape=(256,256,3)):
    """
    generate network based on unet
    """    
        
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    n_filters=32

    img_ch=3 # image channels
    out_ch=n_classes # output channel
    img_size = input_shape
    img_height, img_width = img_size[0], img_size[1]
    padding='same'
    
    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, (k, k), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(pool1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
     
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(pool2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(pool3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)

    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(pool4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)    
    conv5 = Conv2D(16*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    
#upsampling1    
#################################################    
    conv5 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    up1 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
###############################
    conv5 = UpSampling2D(size=(s, s))(conv5)
################################################################
    conv5 = Lambda(squeeze_idt, name='squeeze1_1')(conv5)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    conv5 = Activation('relu')(conv5)
    conv4 = Lambda(squeeze_idt, name='squeeze1_2')(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv4])
    up12 = Concatenate(axis=3)([conv5, conv4])
    
#up122 is the attention module -20190423
    up122 = GlobalAveragePooling2D()(up12)
    up122 = Activation('sigmoid')(Multiply()([up12,up122]))
#    up12 = up12*up122
##################################################
###############################################    
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
###################################
    conv6 = Activation('relu')(conv6)
    conv6 = Add()([conv6, up122])

    
#upsampling2      
###########################################    
    conv6 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
##################################
    conv6 = UpSampling2D(size=(s, s))(conv6)
############################################################################
    conv6 = Lambda(squeeze_idt, name='squeeze2_1')(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)
    conv3 = Lambda(squeeze_idt, name='squeeze2_2')(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv3])
    up22 = Concatenate(axis=3)([conv6, conv3])
#up122 is the attention module -20190423
    up222 = GlobalAveragePooling2D()(up22)
    up222 = Activation('sigmoid')(Multiply()([up22,up222]))
#    up22 = up22*up222
###################################################################

    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
##############################################

    conv7 = Activation('relu')(conv7) 
    conv7 = Add()([conv7, up222])

    
#upsampling3      
#######################################################################    
    conv7 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)   
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
#####################################################################    
    conv7 = UpSampling2D(size=(s, s))(conv7)
#############################################################
    conv7 = Lambda(squeeze_idt, name='squeeze3_1')(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)
    conv2 = Lambda(squeeze_idt, name='squeeze3_2')(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv3])
    up32 = Concatenate(axis=3)([conv7, conv2])   
#up122 is the attention module -20190423
    up322 = GlobalAveragePooling2D()(up32)
    up322 = Activation('sigmoid')(Multiply()([up32,up322]))
#    up32 = up32*up322
###########################################################################    

    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
##########################################################################

    conv8 = Activation('relu')(conv8)
    conv8 = Add()([conv8, up322])

    
    
#upsampling4  
##############################################################
    conv8 = Conv2D(n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
############################################################################    
    conv8 = UpSampling2D(size=(s, s))(conv8)
    
    conv8 = Lambda(squeeze_idt, name='squeeze4_1')(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    conv1 = Lambda(squeeze_idt, name='squeeze4_2')(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)
    
#    up12 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv5), conv3])
    up42 = Concatenate(axis=3)([conv8, conv1])
#up122 is the attention module -20190423
    up422 = GlobalAveragePooling2D()(up42)
    up422 = Activation('sigmoid')(Multiply()([up42,up422]))
#    up42 = up42*up422
################################################################################    
    
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)    
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    

    conv9 = Activation('relu')(conv9)
    conv9 = Add()([conv9, up422])
    
    # this is for alignment in the value of input [-1, 1]
    #outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='sigmoid')(conv9)
    outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='tanh')(conv9)

    g = Model(inputs, outputs)
    
#    g = multi_gpu_model(g, gpus=2 )

    return g
'''

def deep_Unet(n_classes=3, input_shape=(256,256,3)):
    """
    generate network based on unet
    """    
        
    # set image specifics
    k=3 # kernel size
    s=2 # stride
    img_ch=1 # image channels
    out_ch=n_classes # output channel
    img_size = input_shape
    n_filters=32
    img_height, img_width = img_size[0], img_size[1]
    padding='same'
    de_k=2
    de_s=2
    
    inputs = Input((img_height, img_width, img_ch))
    conv1 = Conv2D(n_filters, kernel_size=(k, k), strides=(s, s), padding=padding)(inputs)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    #conv1 = Activation('relu')(conv1) 
    a1 = LeakyReLU(0.2)(conv1)  

    '''
    conv1 = Conv2D(n_filters, (k, k),  padding=padding)(conv1)
    conv1 = BatchNormalization(scale=False, axis=3)(conv1)
    conv1 = Activation('relu')(conv1)    
    pool1 = MaxPooling2D(pool_size=(s, s))(conv1)
    '''

    conv2 = Conv2D(2*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a1)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    #conv2 = Activation('relu')(conv2) 
    a2 = LeakyReLU(0.2)(conv2) 

    '''
    conv2 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv2)
    conv2 = BatchNormalization(scale=False, axis=3)(conv2)
    conv2 = Activation('relu')(conv2)    
    pool2 = MaxPooling2D(pool_size=(s, s))(conv2)
    '''

    conv3 = Conv2D(4*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a2)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    #conv3 = Activation('relu')(conv3)
    a3 = LeakyReLU(0.2)(conv3) 

    '''
    conv3 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv3)
    conv3 = BatchNormalization(scale=False, axis=3)(conv3)
    conv3 = Activation('relu')(conv3)    
    pool3 = MaxPooling2D(pool_size=(s, s))(conv3)
    '''

    conv4 = Conv2D(8*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a3)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    #conv4 = Activation('relu')(conv4) 
    a4 = LeakyReLU(0.2)(conv4) 

    '''   
    conv4 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv4)
    conv4 = BatchNormalization(scale=False, axis=3)(conv4)
    conv4 = Activation('relu')(conv4)    
    pool4 = MaxPooling2D(pool_size=(s, s))(conv4)
    '''

    conv5 = Conv2D(8*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a4)
    conv5 = BatchNormalization(scale=False, axis=3)(conv5)
    a5 = LeakyReLU(0.2)(conv5) 


    conv6 = Conv2D(8*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a5)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    a6 = LeakyReLU(0.2)(conv6) 

    conv7 = Conv2D(8*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a6)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    a7 = LeakyReLU(0.2)(conv7) 

    conv8 = Conv2D(8*n_filters, kernel_size=(k, k),  strides=(s, s), padding=padding)(a7)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    a8 = LeakyReLU(0.2)(conv8) 

    conv9 = Conv2D(8*n_filters, kernel_size=(2, 2),  strides=(1, 1), padding='valid')(a8)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    a9 = LeakyReLU(0.2)(conv9) 
    
    ############################################deconvolution starts

    dconv1 = Conv2DTranspose(8*n_filters, kernel_size=(2, 2), strides=(1, 1), padding='same')(a9)
    dconv1 = BatchNormalization(scale=False, axis=3)(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    
    up1 = Concatenate(axis=3)([dconv1, conv8])

    up1 = LeakyReLU(0.2)(up1)


    dconv2 = Conv2DTranspose(8*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up1)
    dconv2 = BatchNormalization(scale=False, axis=3)(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    
    up2 = Concatenate(axis=3)([dconv2, conv7])
    up2 = LeakyReLU(0.2)(up2)

    dconv3 = Conv2DTranspose(8*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up2)
    dconv3 = BatchNormalization(scale=False, axis=3)(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    
    up3 = Concatenate(axis=3)([dconv3, conv6])
    up3 = LeakyReLU(0.2)(up3)

    dconv4 = Conv2DTranspose(8*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up3)
    dconv4 = BatchNormalization(scale=False, axis=3)(dconv4)
    #dconv4 = Dropout(0.5)(dconv4)
    
    up4 = Concatenate(axis=3)([dconv4, conv5])
    up4 = LeakyReLU(0.2)(up4)

    dconv5 = Conv2DTranspose(8*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up4)
    dconv5 = BatchNormalization(scale=False, axis=3)(dconv5)
    #dconv5 = Dropout(0.5)(dconv5)
    
    up5 = Concatenate(axis=3)([dconv5, conv4])
    up5 = LeakyReLU(0.2)(up5)

    dconv6 = Conv2DTranspose(4*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up5)
    dconv6 = BatchNormalization(scale=False, axis=3)(dconv6)
    #dconv6 = Dropout(0.5)(dconv6)
    
    up6 = Concatenate(axis=3)([dconv6, conv3])
    up6 = LeakyReLU(0.2)(up6)

    dconv7 = Conv2DTranspose(2*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up6)
    dconv7 = BatchNormalization(scale=False, axis=3)(dconv7)
    #dconv7 = Dropout(0.5)(dconv7)
    
    up7 = Concatenate(axis=3)([dconv7, conv2])
    up7 = LeakyReLU(0.2)(up7)

    dconv8 = Conv2DTranspose(1*n_filters, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up7)
    dconv8 = BatchNormalization(scale=False, axis=3)(dconv8)
    #dconv7 = Dropout(0.5)(dconv7)
    
    up8 = Concatenate(axis=3)([dconv8, conv1])
    up8 = LeakyReLU(0.2)(up8)

    dconv9 = Conv2DTranspose(out_ch, kernel_size=(de_k, de_k), strides=(de_s, de_s), padding='same')(up8)


    '''
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(up1)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
    conv6 = Conv2D(8*n_filters, (k, k),  padding=padding)(conv6)
    conv6 = BatchNormalization(scale=False, axis=3)(conv6)
    conv6 = Activation('relu')(conv6)    
     
    up2 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv6), conv3])
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(up2)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    conv7 = Conv2D(4*n_filters, (k, k),  padding=padding)(conv7)
    conv7 = BatchNormalization(scale=False, axis=3)(conv7)
    conv7 = Activation('relu')(conv7)    
    
    up3 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv7), conv2])
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(up3)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)    
    conv8 = Conv2D(2*n_filters, (k, k),  padding=padding)(conv8)
    conv8 = BatchNormalization(scale=False, axis=3)(conv8)
    conv8 = Activation('relu')(conv8)
    
    up4 = Concatenate(axis=3)([UpSampling2D(size=(s, s))(conv8), conv1])
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(up4)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)    
    conv9 = Conv2D(n_filters, (k, k),  padding=padding)(conv9)
    conv9 = BatchNormalization(scale=False, axis=3)(conv9)
    conv9 = Activation('relu')(conv9)
    '''
    
    #outputs = Conv2D(out_ch, (1, 1), padding=padding, activation='tanh')(dconv9)
    outputs = Activation('softmax')(dconv9)
    
    g = Model(inputs, outputs)

    return g



def InceptionV3Model_fl(n_classes=2, input_shape=(299, 299, 3)):
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='global_dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='global_dense2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model


def InceptionV3Model(n_classes=2, input_shape=(299, 299, 3)):
    base_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model


def DensenetModel(n_classes=2, input_shape=(224, 224, 3)):
    base_model = densenet.DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('relu').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model

def DensenetModel_fl(n_classes=2, input_shape=(224, 224, 3)):
    base_model = densenet.DenseNet169(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('relu').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='global_dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='global_dense2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model

def NasnetModel(n_classes=2, input_shape=(331, 331, 3)):
    base_model = nasnet.NASNetLarge(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('activation_260').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model
    
def InceptionResNetV2Model(n_classes=2, input_shape=(299,299,3)):
    base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_7b_ac').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model

def InceptionResNetV2Model_fl(n_classes=2, input_shape=(299,299,3)):
    base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv_7b_ac').output)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='global_dense1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='global_dense2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
    final_model = Model(inputs=model.input, outputs=predictions)
    return final_model

def Inceptionv3_cam(n_classes=2, input_shape=(299,299,3)):
    base_model = inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=True, input_shape=input_shape)
    
    return base_model

