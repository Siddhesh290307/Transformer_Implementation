from model.encoder import Encoder
from model.decoder import Decoder
from model.positional_encoding import positional_encoding
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Embedding

class Transformer(tf.keras.Model):

    def __init__(self, N, H, d_model, dk, dv, dff,
                 src_vocab_size, tgt_vocab_size,
                 max_positional_encoding,
                 dropout_rate=0.1, layernorm_eps=1e-6):

        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)

        self.PE = positional_encoding(max_positional_encoding, d_model)

        self.dropout_enc = Dropout(dropout_rate)
        self.dropout_dec = Dropout(dropout_rate)

        self.encoder = Encoder(
            N, H, d_model, dk, dv, dff,
            dropout_rate, layernorm_eps
        )

        self.decoder = Decoder(
            N, H, d_model, dk, dv, dff,
            dropout_rate, layernorm_eps
        )

        self.fc_out = Dense(tgt_vocab_size)

    def call(self, x, y, training=False,
             enc_padding_mask=None,
             look_ahead_mask=None,
             dec_padding_mask=None):

        
        # Encoder
        seq_len_x = tf.shape(x)[1]

        x = self.src_embedding(x)                     
        x += self.PE[:, :seq_len_x, :]
        x = self.dropout_enc(x, training=training)

        enc_output = self.encoder(
            x,
            training=training,
            mask=enc_padding_mask
        )

        #Decoder
        seq_len_y = tf.shape(y)[1]

        y = self.tgt_embedding(y)                    
        y += self.PE[:, :seq_len_y, :]
        y = self.dropout_dec(y, training=training)

        dec_output = self.decoder(
            y,
            enc_output,
            training=training,
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask
        )

        return self.fc_out(dec_output)
