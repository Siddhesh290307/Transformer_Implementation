import tensorflow as tf
import numpy as np


#Performing the operation: Attention(Q,K,V)= softmax(Q*K^T/sqrt(d_k))*V
def scaled_dot_product_attention(Q, K, V, mask=None):
    matmul_QK = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_logits = matmul_QK / tf.math.sqrt(dk)

    if mask is not None:
        scaled_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_logits, axis=-1)
    output = tf.matmul(attention_weights, V)

    return output



#A single Multihead Attention Layer
class Multihead_Attention(tf.keras.layers.Layer):
    def __init__(self, H, d_model, dk, dv):    
        super(Multihead_Attention, self).__init__()
        
        #Helps keep variance stable through layers
        initializer= tf.keras.initializers.GlorotUniform()

        #Per-head projection matrices
        self.WQ =tf.Variable(initializer(shape=(H, d_model, dk)), trainable=True)
        self.WK =tf.Variable(initializer(shape=(H, d_model, dk)), trainable=True)
        self.WV =tf.Variable(initializer(shape=(H, d_model, dv)), trainable=True)

        #Output projection after concatenating the heads
        self.WO =tf.Variable(initializer(shape=(H * dv, d_model)), trainable=True)

    
    def call(self, Q, K, V, mask=None):
        
        Qh = tf.einsum('btd,hdk->bhtk', Q, self.WQ)
        Kh = tf.einsum('btd,hdk->bhtk', K, self.WK)
        Vh = tf.einsum('btd,hdv->bhtv', V, self.WV)

        Ah = scaled_dot_product_attention(Qh, Kh, Vh, mask)

        Ah = tf.transpose(Ah, [0, 2, 1, 3])
        A = tf.reshape(Ah, (tf.shape(Ah)[0], tf.shape(Ah)[1], -1))

        A = tf.matmul(A, self.WO)

        
        return A
