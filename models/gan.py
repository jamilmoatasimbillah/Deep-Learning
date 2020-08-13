from tensorflow import keras


def create(generator, discriminator, name="GAN Model", summary=False):
    discriminator.trainable = False
    inputs = keras.layers.Input(shape=(512,512,3))
    
    fake = generator(inputs)
    outputs = discriminator(fake)
    gan_model = keras.models.Model(inputs=inputs, outputs=[fake, outputs], name=name)
    
    None if not summary else gan_model.summary()
    discriminator.trainable = True
    return gan_model