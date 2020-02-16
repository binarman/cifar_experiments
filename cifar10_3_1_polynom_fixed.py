import cifar10_base
import keras

class Polynom(keras.layers.Layer):
  def _init_(self, **kwargs):
    super(Polynom, self)._init_(**kwargs)

  def build(self, input_shape):
    super(Polynom, self).build(input_shape)

  def call(self, x):
    return keras.layers.multiply([x, x, x]) + x

  def compute_output_shape(self, input_shape):
    return input_shape

if __name__ == "__main__":
  cifar10_base.compute_with_activation(Polynom)
