import cifar10_base
import keras

class FusedActivation(keras.layers.Layer):
  def _init_(self, **kwargs):
    super(FusedActivation, self)._init_(**kwargs)

  def build(self, input_shape):
    self.coeff1 = self.add_weight(name="coeff1",
                                  shape=(1,),
                                  initializer=keras.initializers.Constant(value=1),
                                  trainable=True)
    self.coeff2 = self.add_weight(name="coeff2",
                                  shape=(1,),
                                  initializer=keras.initializers.Constant(value=1),
                                  trainable=True)
    self.coeff3 = self.add_weight(name="coeff3",
                                  shape=(1,),
                                  initializer=keras.initializers.Constant(value=1),
                                  trainable=True)
    super(FusedActivation, self).build(input_shape)

  def call(self, x):
    pol = keras.layers.multiply([x, x, x, self.coeff3]) + keras.layers.multiply([x,x,self.coeff2]) + keras.layers.multiply([x,self.coeff1])
    return keras.activations.relu(pol)

  def compute_output_shape(self, input_shape):
    return input_shape

if __name__ == "__main__":
  cifar10_base.compute_with_activation(FusedActivation)
