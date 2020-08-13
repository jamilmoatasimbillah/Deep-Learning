import tensorflow as tf
from random import shuffle
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import generator
import discriminator
import gan
from helpers import get_random_crop



class GAN:
    def __init__(self, loss_fn, generator_name, discriminator_name, *kargs, **kwargs):
        self._generator = generator.create(loss_fn)
        self._discriminator = discriminator.create()
        self._gan = gan.create(self.generator, self.discriminator)
        
        self.generator_name = generator_name
        self.discriminator_name = discriminator_name
    
    
    def compile(self, *kargs, **kwargs):
        self._gan.compile(*kargs, **kwargs)
    
    
    def load_weights(self, generator_file, discriminator_file):
        self._generator.load_weights(generator_file)
        self._discriminator.load_weights(discriminator_file)
        
    
    def fit(self, imgs, 
                    epochs=1, batch_size=10, 
                    train_disc = True):

        n = len(imgs)
        for epoch in range(epochs):
            shuffle(imgs)
            t = tqdm(range(0, n, batch_size))
            t.desc = f"Epoch {epoch+1}/{epochs} -"
            d_loss, accuracy = 0, 0
            for i in t:            
                Y = [get_random_crop(img, 512, 512) for img in imgs[i:i+batch_size]]
                X = tf.keras.backend.variable([cv2.resize(cv2.resize(img, (128,128)), (512,512), cv2.INTER_CUBIC) for img in Y])
                d_loss, accuracy, g_loss = self.train_on_batch(X, Y, train_disc)
                t.postfix = f"d_loss: {d_loss:.5f} a: {accuracy*100:.2f}% g_loss: [{g_loss[0]:.5f} {g_loss[1]:.5f} {g_loss[2]:.5f}] "
            self.save()
    
            
    def save(self):
        self._discriminator.save(f"{self.discriminator_name}_4x.h5")
        self._generator.save(f"{self.generator_name}_4x.h5")
    
    
    def summary(self, gan=True, generator=False, discriminator=False):
        None if generator else self._generator.summary()
        None if discriminator else self._discriminator.summary()
        self.gan.summary()
    
    
    def _train_discriminator(self, real, fake):
        self._discriminator.trainable = True
        ones, zeros = np.ones(len(real)), np.zeros(len(fake))
        outputs = np.concatenate((ones, zeros))
        inputs = np.concatenate((real, fake))
        random_shuffle_index = np.random.permutation(len(outputs))
        outputs, inputs = outputs[random_shuffle_index], inputs[random_shuffle_index]
        loss, accuracy = self._discriminator.train_on_batch(inputs, outputs)
        self._discriminator.trainable = False
        return loss, accuracy
    
    
    def train_on_batch(self,  inputs, outputs, train_disc=True):
        d_loss, accuracy = 0, 0
        if train_disc:
            fake = self.generator.predict_on_batch(inputs)
            self._discriminator.trainable = True
            d_loss, accuracy = self._train_discriminator(outputs, fake)
            
        self._discriminator.trainable = False
        g_loss = self.gan.train_on_batch(inputs, [tf.keras.backend.variable(outputs), np.ones(len(outputs)) ])

        return d_loss, accuracy, g_loss
    
    

#############################################################################################################

class SuperResolutor:
    def __init__(self, *kargs, **kwargs):
        self._generator = generator.create()
        
    def low_to_up(self, imgs, batch_size=10, filepath=False):
        n = len(imgs)
        for i in tqdm(range(0,n, batch_size)):
            X = imgs[i*batch_size:(i+1)*batch_size]
            if filepath:
                X = [plt.imread(img) for img in X]
            
            self._generator.predict_on_batch()




































