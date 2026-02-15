import numpy as np
import tensorflow as tf

#positions is the maximum sequence length

def positional_encoding(positions, d):

    pos = np.arange(positions)[:, np.newaxis]

    k = np.arange(d)[np.newaxis, :]
    #Each row corresponds to 1 token position
    #Each column corresponds to one embedding dimension.
    i = k//2

    #angle_rads = pos / (10000 ** (2 * i / d))
    angle_rads = pos/(10000 ** (2*i/d))

    #sine for even dimensions
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    #cosine for odd dimensions
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)