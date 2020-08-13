import tensorflow as tf
from helpers import conv2d, deconv2d


def get_loss_fn(input_shape=(None, None, 3)):
    
    vgg19 = tf.keras.applications.VGG19(input_shape=input_shape, include_top=False, weights="imagenet")
    loss_model = tf.keras.models.Model(vgg19.input, vgg19.get_layer('block5_conv4').output)
    
    @tf.function
    def loss_fn(y_target, y_predicted):    
        vgg_loss = tf.keras.metrics.MSE(loss_model(y_target), loss_model(y_predicted))
        mse_loss = tf.keras.metrics.MSE(y_target, y_predicted)
        loss = 10*tf.keras.backend.mean(vgg_loss) + tf.keras.backend.mean(mse_loss)
        return loss

    return loss_fn


def create(loss_fn=None, input_shape = ( None, None, 3 ), summary=False, name="generator"):
    
    inputs = tf.keras.layers.Input(shape=input_shape, name="inputs")
    
    d_sample1 = conv2d(inputs, 32, "d_sample1", (3,3), bn=False)
    d_sample2 = conv2d(d_sample1, 64, "d_sample2", (3,3), bn=False)
    d_sample3 = conv2d(d_sample2, 128, "d_sample3", (3,3), bn=False)
    d_sample4 = conv2d(d_sample3, 128, "d_sample4", (3,3), bn=False)

    
    u_sample = deconv2d(d_sample4, d_sample3, 64, "u_sample1", (3,3))
    u_sample = deconv2d(u_sample, d_sample2, 64, "u_sample2", (3,3))
    u_sample = deconv2d(u_sample, d_sample1, 32, "u_sample3", (3,3))


    outputs = deconv2d(u_sample, inputs, 32, "u_sample5", (3,3))
    outputs = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same", name="outputs")(outputs)
    
    model = tf.keras.models.Model(inputs, outputs, name=name)
    None if not summary else model.summary()
    None if not loss_fn else model.compile(optimizer="adam", loss=loss_fn)
    return model
 

