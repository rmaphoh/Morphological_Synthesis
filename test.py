import argparse
import os
import sys
from tensorflow.keras import activations
#import matplotlib.pyplot as plt
import PIL.Image
import tensorflow.keras.backend as K
from matplotlib import pylab as plt
import numpy as np
import matplotlib.cm as cm
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from PIL import Image
import tensorflow as tf

# CUDNN handle error - allow GPU growth
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras_preprocessing import image as keras_image

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

'''
parser=argparse.ArgumentParser()
parser.add_argument(
    '--model_name',
    type=str,
    required=True
    )

parser.add_argument(
    '--image_directory',
    type=str,
    required=True
    )
FLAGS,_= parser.parse_known_args()
'''

def decode_one_hot(one_hot_map):
    return np.argmax(one_hot_map, axis=-1)

def preprocess_image(file_path):
    #img = crop2square(image.load_img(image_path)).resize((512, 512))
    image_1 = image.load_img(file_path).resize((256, 256))
    image_1 = image.img_to_array(image_1)
    image_1 = np.expand_dims(image_1, axis=0)

    #print(np.unique(x))
    image_1 /= 127.5
    image_1 -= 1.

    return image_1


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]
    
    if sort:
        filenames = sorted(filenames)
    
    return filenames



def save_image(image, grayscale = True, title=''):
    '''
    if ax is None:
        plt.figure()
    plt.axis('off')
    '''
    if len(image.shape) == 2 or grayscale == True:
        if len(image.shape) == 3:
            image = np.sum(np.abs(image), axis=2)
        #
        
        plt.imsave('./test_image/'+title+'.jpg', image, cmap=plt.cm.gray, vmin=0, vmax=1)
    else:
        #image = image + 127.5

        image = image.astype('uint8')
        
        plt.imsave('./test_image/'+title+'.jpg', image)
    return


if __name__ == '__main__':

    model_trained = models.load_model('./checkpoints/Unet/weights.05-0.02-1.00.h5', custom_objects={'tf': tf})
    
    #clf.summary()
    image_path = './representation_syn/testing_img'
    #img_files=all_files_under(image_path, extension=".png")
    img_files=all_files_under(image_path)
    for file_index in range(len(img_files)):
        
        x = preprocess_image(img_files[file_index])
        #print(np.unique(x))
        #x = image.img_to_array(x)
        print('shape of input: ', np.shape(x))
        #original = original_image(img_files[file_index])
        #save_image(original, grayscale=False, title='original'+ str(file_index))
        #img = np.squeeze(x)
        #print(np.shape(img))
        score = model_trained.predict(x, verbose=1)
        #score = np.squeeze(score, axis=0)
        score = np.squeeze(score)
        #score = np.squeeze(score)
        print('shape of score======: ', np.shape(score))
        print('shape of output1111111111111111: ', np.unique(score[...,0]))
        print('shape of output2222222222222222: ', np.unique(score[...,1]))
        print('shape of output3333333333333333: ', np.unique(score[...,2]))
        #print('shape of output4444444444444444: ', np.unique(score[...,3]))
        score_max = decode_one_hot(score)
        print(np.unique(score_max))
        print(np.shape(score_max))

        image_match = np.zeros((256,256,3))
        image_match_0 = image_match[...,0]
        image_match_1 = image_match[...,1]
        image_match_2 = image_match[...,2]
        image_match_0[score_max==1]=255
        #image_match_1[score_max==0]=255
        image_match_2[score_max==2]=255

        r_img = Image.fromarray(image_match_0).convert('L')
        print('=======', np.shape(r_img))
        g_img = Image.fromarray(image_match_1).convert('L')
        b_img = Image.fromarray(image_match_2).convert('L')



        pic = Image.merge('RGB',(r_img,g_img,b_img))  # 合并通道
        plt.imshow(pic),plt.axis('off')
        plt.axis('off') 
        #plt.show()

        '''
        #image_map = PIL.Image.fromarray(score).convert('L')
        #image_map = PIL.Image.fromarray(image_match)
        print(np.shape(image_map))
        #plt.imshow(revised_label_image, cmap='gray'),plt.axis('off')
        #plt.axis('off') 
        print('final',np.shape(image_map))
        print(type(image_map))
        print(np.unique(image_map))
        
        #plt.imshow(image_map, cmap='gray'),plt.axis('off')
        #plt.axis('off') 
        #plt.show()
        '''
        pic.save('./test_image/test_{:02}.png'.format(file_index+1))

        
    
    

