from random import shuffle
import numpy as np
from tqdm import tqdm
from tensorflow import keras



def train_discriminator(discriminator, real, fake):
    discriminator.trainable = True
    ones, zeros = np.ones(len(real)), np.zeros(len(fake))
    outputs = np.concatenate((ones, zeros))
    inputs = np.concatenate((real, fake))
    random_shuffle_index = np.random.permutation(len(outputs))
    outputs, inputs = outputs[random_shuffle_index], inputs[random_shuffle_index]
    loss, accuracy = discriminator.train_on_batch(inputs, outputs)
    discriminator.trainable = False
    return loss, accuracy


def train_gan(gan, generator, discriminator, 
              imgs, epochs=1, batch_size=10, 
              train_disc = True,
              generator_name="model_generator", 
              discriminator_name="model_discriminator"):
    n = len(imgs)    
    for epoch in range(epochs):
        shuffle(imgs)
        t = tqdm(range(0, n, batch_size))
        t.desc = f"Epoch {epoch+1}/{epochs} -"
        d_loss, accuracy = 0, 0
        for i in t:            
            real = [get_random_crop(img, 512, 512) for img in imgs[i:i+batch_size]]
            inputs = keras.backend.variable([cv2.resize(cv2.resize(img, (128,128)), (512,512), cv2.INTER_CUBIC) for img in real])
            if train_disc:
                fake = generator.predict_on_batch(inputs)
                discriminator.trainable = True
                d_loss, accuracy = train_discriminator(discriminator, real, fake)
            discriminator.trainable = False
 
            g_loss = gan.train_on_batch(inputs, [tf.keras.backend.variable(real), np.ones(len(real)) ])
            t.postfix = f"d_loss: {d_loss:.5f} a: {accuracy*100:.2f}% g_loss: [{g_loss[0]:.5f} {g_loss[1]:.5f} {g_loss[2]:.5f}] "
            
        discriminator.save(f"./drive/My Drive/Image SR/2/{discriminator_name}_4x_gan.h5")
        generator.save(f"./drive/My Drive/Image SR/2/{generator_name}_4x_gan.h5")