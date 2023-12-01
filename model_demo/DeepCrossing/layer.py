import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, Attention
import tensorflow.keras.backend as K
import operators as sgx


class MyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.units = units
        self.w = None
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weight", shape=(input_shape[1], self.units), trainable=True
        )
        super(MyDense, self).build(input_shape)

    def call(self, inputs):
        if self.activation == None:
            return sgx.e_matmul(inputs, self.w)
        elif self.activation == "relu":
            return sgx.e_relu(sgx.e_matmul(inputs, self.w))
        elif self.activation == "sigmoid":
            return sgx.e_sigmoid(sgx.e_matmul(inputs, self.w))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(MyDense, self).get_config()
        config.update({"units": self.units})
        config.update({"activation": self.activation})
        return config


class Embed_layer(Layer):
    def __init__(self, k, sparse_feature_columns):
        super(Embed_layer, self).__init__()
        self.emb_layers = [
            Embedding(feat["feat_onehot_dim"], k) for feat in sparse_feature_columns
        ]

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("The dim of inputs should be 2, not %d" % (K.ndim(inputs)))
        emb = tf.transpose(
            tf.convert_to_tensor(
                [layer(inputs[:, i]) for i, layer in enumerate(self.emb_layers)]
            ),
            [1, 0, 2],
        )
        emb = tf.reshape(emb, shape=(-1, emb.shape[1] * emb.shape[2]))
        return emb


class Res_layer(Layer):
    def __init__(self, hidden_units):
        super(Res_layer, self).__init__()
        self.dense_layer = [MyDense(i, activation="relu") for i in hidden_units]

    def build(self, input_shape):
        self.output_layer = MyDense(input_shape[-1], activation=None)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("The dim of inputs should be 2, not %d" % (K.ndim(inputs)))
        x = inputs
        for layer in self.dense_layer:
            x = layer(x)
        x = self.output_layer(x)
        output = inputs + x
        return sgx.e_relu(output)
