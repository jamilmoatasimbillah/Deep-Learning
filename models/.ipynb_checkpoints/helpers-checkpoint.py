from tensorflow import keras
import numpy as np

def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop

def conv2d(layer_input, filters, name, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same', name = name+'_c')(layer_input)
        d = keras.layers.LeakyReLU(alpha=0.2, name = name+'_a')(d)
        if bn:
            d = keras.layers.BatchNormalization(momentum=0.8, name = name+'_bn')(d)
        return d

def deconv2d(layer_input,  skip_input, filters, name, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    u = keras.layers.UpSampling2D(size=2, name = name+'_up')(layer_input)
    u = keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu',name = name+'_c')(u)
    if dropout_rate:
        u = keras.layers.Dropout(dropout_rate, name = name+'_dout')(u)
    #u = keras.layers.BatchNormalization(momentum=0.8, name = name+'_bn')(u)
    u = keras.layers.Concatenate( name = name+'_con')([u, skip_input])
    return u