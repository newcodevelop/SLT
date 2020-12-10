# -*- coding: utf-8 -*-
"""Interactive SLT-1.ipynb



##### Copyright 2019 Dibyanayan Bandyopadhyay
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

#!pip install -q tfds-nightly


#!pip install matplotlib==3.2.2

#!pip install -q tqdm

#from google.colab import drive

#drive.mount('/content/gdrive', force_remount=True)


import tensorflow as tf
"""
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	for i in physical_devices:
		tf.config.experimental.set_memory_growth(i, True)
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kerastuner as kt
# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm
import glob
import shutil
import os
import argparse
import cv2
import argparse
import nltk.translate.bleu_score as bleu
import sacrebleu
my_parser = argparse.ArgumentParser()
my_parser.version = '1.0'

my_parser.add_argument('-BATCH_SIZE', action='store', type=int, default=64)
my_parser.add_argument('-BUFFER_SIZE', action='store', type=int, default=10)
my_parser.add_argument('-train_size', action='store', type=int, default=128)
my_parser.add_argument('-do_test_on_train', action='store', type=int, default=0)

my_parser.add_argument('-num_layers_enc', action='store', type=int, default=2)
my_parser.add_argument('-num_layers_dec', action='store', type=int, default=2)
my_parser.add_argument('-d_model', action='store', type=int, default=512)
my_parser.add_argument('-dff', action='store', type=int, default=512)
my_parser.add_argument('-num_heads', action='store', type=int, default=8)
my_parser.add_argument('-EPOCHS', action='store', type=int, default=40)

my_parser.add_argument('-train_dir', action='store', type=str, required=True)
my_parser.add_argument('-test_dir', action='store', type=str, required=True)
args = my_parser.parse_args()


BATCH_SIZE = args.BATCH_SIZE
BUFFER_SIZE = args.BUFFER_SIZE
train_size = args.train_size
num_layers_enc = args.num_layers_enc
num_layers_dec = args.num_layers_dec
d_model = args.d_model
dff = args.dff
num_heads = args.num_heads
EPOCHS = args.EPOCHS
dropout_rate = 0.1
do_test_on_train = args.do_test_on_train


import pickle
import gzip
def load_dataset_file(filename):
  with gzip.open(filename, "rb") as f:
    loaded_object = pickle.load(f)
    return loaded_object

p = load_dataset_file(args.train_dir)

videos,captions,glosses = [],[],[]
for i in range(len(p)):
  videos.append(np.asarray(p[i]['sign']))
  captions.append(p[i]['text'])
  glosses.append(p[i]['gloss'])



def get_tokenizer(arr,for_what='',start_token="",end_token=""):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE

    tokenizer = Tokenizer(BPE())
    from tokenizers.trainers import BpeTrainer

    trainer = BpeTrainer(special_tokens=["[PAD]","[UNK]", "[CLS]", "[SEP]", "[MASK]", start_token, end_token])
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer.pre_tokenizer = Whitespace()
    train_captions = [i+'\n' for i in arr]
    f1 = open('train_{}_dir.txt'.format(for_what), 'w')
    f1.writelines(train_captions)
    f1.close()
    tokenizer.train(trainer, [os.path.join(os.getcwd(),'train_{}_dir.txt'.format(for_what))])
    from tokenizers.processors import TemplateProcessing

    tokenizer.post_processor = TemplateProcessing(
        single="{} $A {}".format(start_token,end_token),
        special_tokens=[
            (start_token, tokenizer.token_to_id(start_token)),
            (end_token, tokenizer.token_to_id(end_token)),
        ],
    )
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    o = tokenizer.encode_batch(arr)
    train_captions_final = np.asarray([i.ids for i in o],dtype = np.int32)
    return train_captions_final,tokenizer

train_captions_final,tokenizer = get_tokenizer(captions,'translation',"[TRANS-START]","[TRANS-END]") 
train_glosses_final,tokenizer_r = get_tokenizer(glosses,'recognition',"[RECOG-START]","[RECOG-END]") 


l = np.shape(train_captions_final)[-1]
train_glosses_final = tf.keras.preprocessing.sequence.pad_sequences(
    train_glosses_final, maxlen=l, dtype='int32', padding='post'
)

assert np.shape(train_captions_final)[-1]==np.shape(train_glosses_final)[-1]
print('captions shape {}'.format(np.shape(train_captions_final)))
print('glosses shape {}'.format(np.shape(train_glosses_final)))

def create_padding_mask_image(videos):
  
  lens = []
  for i in videos:
	  lens.append(i.shape[0])
  m_len = max(lens)
  print(m_len)
  mask=[]
  for i in videos:
	  l = i.shape[0]
	  temp1 = [0 for _ in range(l)]
	  temp2 = [1 for _ in range(m_len-l)]
	  mask.append(temp1+temp2)
  mask = tf.cast(mask,tf.float32)
  print('done')
  #return mask[:, tf.newaxis, tf.newaxis, :]
  videos = tf.keras.preprocessing.sequence.pad_sequences(videos, padding='post', dtype = 'float32')
  return np.asarray(mask),videos
video_mask,videos = create_padding_mask_image(videos)

print(
  'DONE UPTO THIS'
)
masks = []
for vid in videos:
  mask = []
  for i in vid:
    #print(i)
    if (np.asarray(i)==np.zeros(len(i))).all():
      mask.append(1)
    else:
      mask.append(0)
  masks.append(mask)


assert (video_mask==masks).all()

videos = [i for i in videos]


for i,j in zip(train_captions_final,captions):
  assert tokenizer.decode(i)==j
"""
for i,j in zip(train_glosses_final,glosses):
  print((tokenizer_r.decode(i),j))
  assert tokenizer_r.decode(i)==j
"""
def map_func(img_name, cap,gloss,mask):
  #img_tensor = np.load(os.path.join(os.getcwd(),'processed_img',img_name.decode('utf-8').split('/')[-1]+'_constructed'+'.npy'))
  
  return img_name, cap,gloss,mask


def tf_encode(img, cap,gloss,mask):
  result_pt, result_en, result_en1,mask = tf.numpy_function(map_func, [img, cap,gloss,mask], [tf.float32, tf.int32, tf.int32,tf.float32])
  result_en.set_shape([None])
  result_en1.set_shape([None])
  result_pt.set_shape([None,None])
  mask.set_shape([None])
  return result_pt, result_en, result_en1,mask

dataset = tf.data.Dataset.from_tensor_slices((videos, train_captions_final,train_glosses_final,video_mask))

# Use map to load the numpy files in parallel
dataset = dataset.map(tf_encode)

# Shuffle and batch


dataset = dataset.shuffle(BUFFER_SIZE)




print('dataset created')



input_vocab_size = tokenizer.get_vocab_size()  # useless for image
target_vocab_size_recog = tokenizer_r.get_vocab_size() 
target_vocab_size_trans = tokenizer.get_vocab_size() 

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

"""## Masking

Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. The mask indicates where pad value `0` is present: it outputs a `1` at those locations, and a `0` otherwise.
"""

def create_padding_mask_text(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


"""The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.

This means that to predict the third word, only the first and second word will be used. Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
"""



def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights



class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    #print('d_model {}'.format(d_model))
    #print('num_heads {}'.format(self.num_heads))
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])



class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2



class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    return out3, attn_weights_block1, attn_weights_block2

"""### Encoder

The `Encoder` consists of:
1.   Input Embedding
2.   Positional Encoding
3.   N encoder layers

The input is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the encoder layers. The output of the encoder is the input to the decoder.
"""

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    #self.embedding = tf.keras.layers.Dense(d_model)
    self.embedding = tf.keras.layers.Dense(d_model,activation='relu')
    self.normalization = tf.keras.layers.BatchNormalization()
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x = self.normalization(x,training = training)
    
    #print(np.shape(x))
    #x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)
    
    return x  # (batch_size, input_seq_len, d_model)

"""### Decoder

The `Decoder` consists of:
1.   Output Embedding
2.   Positional Encoding
3.   N decoder layers

The target is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the decoder layers. The output of the decoder is the input to the final linear layer.
"""

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.dec_layer_trans = DecoderLayer(d_model, num_heads, dff, rate)
    self.dec_layer_recog = DecoderLayer(d_model, num_heads, dff, rate)
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, 
           look_ahead_mask_recog, padding_mask_recog,look_ahead_mask_trans, padding_mask_trans):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask_trans, padding_mask_trans)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

    x_trans,_,_ = self.dec_layer_trans(x,enc_output,training,look_ahead_mask_trans,padding_mask_trans)
    x_recog,_,_ = self.dec_layer_recog(x,enc_output,training,look_ahead_mask_recog,padding_mask_recog)
    # x.shape == (batch_size, target_seq_len, d_model)
    return x_recog,x_trans, attention_weights


"""## Create the Transformer

Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the input to the linear layer and its output is returned.
"""



d = {'d_model':d_model,'dff':dff,'ivs':input_vocab_size,'tvsr':target_vocab_size_recog,'tvst':target_vocab_size_trans,'pe_input':input_vocab_size,'pe_target':target_vocab_size_trans,'dr':dropout_rate}
def build_model(hp):
  """Builds a transformer model."""
  

  class Transformer(tf.keras.Model):
    def __init__(self, d_model, dff, input_vocab_size, 
                target_vocab_size_recog,target_vocab_size_trans, pe_input, pe_target, rate=0.1):
      super(Transformer, self).__init__()
      num_layers_enc = hp.Int('num_enc', 2, 4, step=1, default=2)
      num_layers_dec = hp.Int('num_dec', 2, 4, step=1, default=2)
      #num_heads = hp.Choice('num_heads',values=[8,16])
      num_heads = hp.Choice('num_heads',values=[4,8])
      self.encoder = Encoder(num_layers_enc, d_model, num_heads, dff, 
                            input_vocab_size, pe_input, rate)

      self.decoder = Decoder(num_layers_dec, d_model, num_heads, dff, 
                            target_vocab_size_trans, pe_target, rate)

      self.final_layer_recog = tf.keras.layers.Dense(target_vocab_size_recog)
      self.final_layer_trans = tf.keras.layers.Dense(target_vocab_size_trans)
      
    def call(self, inp, tar, training, enc_padding_mask, 
            look_ahead_mask_recog, dec_padding_mask_recog, look_ahead_mask_trans, dec_padding_mask_trans):

      enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
      
      # dec_output.shape == (batch_size, tar_seq_len, d_model)
      dec_output_recog,dec_output_trans, attention_weights = self.decoder(
          tar, enc_output, training, look_ahead_mask_recog, dec_padding_mask_recog, look_ahead_mask_trans, dec_padding_mask_trans)
      
      final_output_recog = self.final_layer_recog(dec_output_recog)  # (batch_size, tar_seq_len, target_vocab_size)
      final_output_trans = self.final_layer_trans(dec_output_trans)
      return final_output_recog,final_output_trans, attention_weights
  model = Transformer(d['d_model'], d['dff'],
                          d['ivs'], d['tvsr'],d['tvst'], 
                          pe_input=d['pe_input'], 
                          pe_target=d['pe_target'],
                          rate=d['dr'])
  
  return model




"""## Set hyperparameters

To keep this example small and relatively fast, the values for *num_layers, d_model, and dff* have been reduced. 

The values used in the base model of transformer were; *num_layers=6*, *d_model = 512*, *dff = 2048*. See the [paper](https://arxiv.org/abs/1706.03762) for all the other versions of the transformer.

Note: By changing the values below, you can get the model that achieved state of the art on many tasks.
"""



"""## Optimizer

Use the Adam optimizer with a custom learning rate scheduler according to the formula in the [paper](https://arxiv.org/abs/1706.03762).

$$\Large{lrate = d_{model}^{-0.5} * min(step{\_}num^{-0.5}, step{\_}num * warmup{\_}steps^{-1.5})}$$
"""

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



"""## Loss and metrics

Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.
"""

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)



"""## Training and checkpointing"""




def create_masks(tar):
  # Encoder padding mask
 
  #enc_padding_mask = create_padding_mask_image(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  #dec_padding_mask = create_padding_mask_image(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask_text(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return combined_mask

"""Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every `n` epochs."""
"""
checkpoint_path = "./checkpoints/train/peymanbatchnormreluHP"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
"""
"""The target is divided into tar_inp and tar_real. tar_inp is passed as an input to the decoder. `tar_real` is that same input shifted by 1: At each location in `tar_input`, `tar_real` contains the  next token that should be predicted.

For example, `sentence` = "SOS A lion in the jungle is sleeping EOS"

`tar_inp` =  "SOS A lion in the jungle is sleeping"

`tar_real` = "A lion in the jungle is sleeping EOS"

The transformer is an auto-regressive model: it makes predictions one part at a time, and uses its output so far to decide what to do next. 

During training this example uses teacher-forcing (like in the [text generation tutorial](./text_generation.ipynb)). Teacher forcing is passing the true output to the next time step regardless of what the model predicts at the current time step.

As the transformer predicts each word, *self-attention* allows it to look at the previous words in the input sequence to better predict the next word.

To prevent the model from peeking at the expected output the model uses a look-ahead mask.
"""


def evaluate(video_test,model):
  
  final_feature = np.asarray(video_test,dtype= np.float32)
  
  encoder_input = tf.reshape(final_feature,(-1,1024))[tf.newaxis,:,:]
  #print('Encoder input in evaluation {}'.format(tf.shape(encoder_input)))
  decoder_input = [tokenizer.token_to_id("[TRANS-START]")]
  output = tf.expand_dims(decoder_input, 0)
  
  enc_padding_mask,dec_padding_mask = None,None
  MAX_LENGTH = 28
  for i in range(MAX_LENGTH):
    combined_mask = create_masks(output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions1,predictions2, attention_weights = model(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions2[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer.token_to_id("[TRANS-END]"):
      return tf.squeeze(output, axis=0), attention_weights
    
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def translate(img,model,plot=''):
  result, attention_weights = evaluate(img,model)
  
  predicted_sentence1 = tokenizer.decode([i for i in result])
  

  #print('Predicted translation: {}'.format(str(predicted_sentence1)))
  
  return predicted_sentence1
  
p1 = load_dataset_file(args.test_dir)

videos_test,captions_test,glosses_test = [],[],[]
for i in range(len(p1)):
    videos_test.append(np.asarray(p1[i]['sign']))
    captions_test.append(p1[i]['text'])
    glosses_test.append(p1[i]['gloss'])



class MyTuner(kt.Tuner):

    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters
        
        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        """
        train_ds = train_ds.batch(
            hp.Int('batch_size', 32, 128, step=32, default=64))
        
        """
        train_ds = train_ds.padded_batch(hp.Int('batch_size', 32, 64, step=16, default=64),([None,None],[None],[None],[None]))
        model = self.hypermodel.build(trial.hyperparameters)
        
        epoch_loss_metric = tf.keras.metrics.Mean()
        
        learning_rate = CustomSchedule(d_model)

        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
            
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        m_len = 475
        k = 1024
        

        @tf.function(input_signature=[tf.TensorSpec(shape=(None,m_len,k), dtype=tf.float32),tf.TensorSpec(shape=(None,None), dtype=tf.int32),tf.TensorSpec(shape=(None,None), dtype=tf.int32),tf.TensorSpec(shape=(None,None), dtype=tf.float32)])
        def train_step(inp, tar1,tar2,mask):
          #tar_inp1 = tar1[:, :-1]
          tar_real1 = tar1[:, 1:]
          tar_inp2 = tar2[:, :-1]
          tar_real2 = tar2[:, 1:]
          
          #print(mask)
          combined_mask_recog = create_masks(tar_inp2)
          combined_mask_trans = create_masks(tar_inp2)
          enc_padding_mask = mask[:,tf.newaxis,tf.newaxis,:]
          dec_padding_mask_recog,dec_padding_mask_trans = mask[:,tf.newaxis,tf.newaxis,:],mask[:,tf.newaxis,tf.newaxis,:]
          with tf.GradientTape() as tape:
            predictions1,predictions2, _ = model(inp, tar_inp2, 
                                        True, 
                                        enc_padding_mask, 
                                        combined_mask_recog, 
                                        dec_padding_mask_recog,
                                        combined_mask_trans,
                                        dec_padding_mask_trans)
            loss1 = loss_function(tar_real1, predictions1)
            #tf.print([tar_real1,predictions1])
            #tf.print([tar_real2,predictions2])
            loss2 = loss_function(tar_real2, predictions2)
            loss = loss1+loss2

          gradients = tape.gradient(loss, model.trainable_variables)    
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
          
          train_loss(loss)

          train_accuracy(tar_real2, predictions2)
        # `self.on_epoch_end` reports results to the `Oracle` and saves the
        # current state of the Model. The other hooks called here only log values
        # for display but can also be overridden. For use cases where there is no
        # natural concept of epoch, you do not have to call any of these hooks. In
        # this case you should instead call `self.oracle.update_trial` and
        # `self.oracle.save_model` manually.
        

        EPOCHS=60
        for epoch in range(EPOCHS+1):
          start = time.time()
          
          train_loss.reset_states()
          train_accuracy.reset_states()
          
          # inp -> portuguese, tar -> english
          for (batch, (inp, tar1,tar2,mask)) in tqdm(enumerate(train_ds)):
            #print(tar)
            if epoch==0 and batch==0:
              print(tar1)
              print(tar2)
            train_step(inp, tar2,tar1,mask)
            
            
            if batch % 50 == 0:
              print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, batch, train_loss.result(), train_accuracy.result()))
              
          
            
          print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))

          print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
          epoch_acc = train_accuracy.result().numpy()
          if epoch%5==0:
            
            ref,hyp1,temp = [],[],[]
            for m,n in tqdm(zip(videos_test,captions_test)):
                #translated1,translated2,translated3 = translate(m,False)
                translated1 = translate(m,model,False)
                
                hyp1.append(re.sub(r'^\s+|\s+$','',re.sub(r'\.', '', translated1)))
                temp.append(re.sub(r'^\s+|\s+$','',re.sub(r'\.', '', n)))
                
            ref= [temp]
                

            score_bleu1 = sacrebleu.corpus_bleu(hyp1,ref)
            print('bleu {}'.format(score_bleu1.score))
            self.on_epoch_end(trial, model, epoch, logs={'epoch_bleu': score_bleu1.score})
        



















"""

#i,j,k = int(p.shape[1]),int(p.shape[2]),int(p.shape[3])



#video_mask_test = create_padding_mask_image(video_name_vector_test)

def evaluate(video_test):
  
  final_feature = np.asarray(video_test,dtype= np.float32)
  
  encoder_input = tf.reshape(final_feature,(-1,1024))[tf.newaxis,:,:]
  print('Encoder input in evaluation {}'.format(tf.shape(encoder_input)))
  decoder_input = [tokenizer.token_to_id("[TRANS-START]")]
  output = tf.expand_dims(decoder_input, 0)
  
  enc_padding_mask,dec_padding_mask = None,None
  MAX_LENGTH = 28
  for i in range(MAX_LENGTH):
    combined_mask = create_masks(output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions1,predictions2, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions2[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == tokenizer.token_to_id("[TRANS-END]"):
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights




def evaluate_beam(video_test,bl):
  final_feature = np.asarray(video_test,dtype= np.float32)
  
  encoder_input = tf.reshape(final_feature,(-1,1024))[tf.newaxis,:,:]
  #print('Encoder input in evaluation {}'.format(tf.shape(encoder_input)))
  
  decoder_input = [tokenizer.token_to_id("[TRANS-START]")]
  output = tf.expand_dims(decoder_input, 0)
  #output = [START]
  
  enc_padding_mask,dec_padding_mask = None,None
  MAX_LENGTH = 20
  
  combined_mask = create_masks(output)

  # predictions.shape == (batch_size, seq_len, vocab_size)
  predictions1,predictions2, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
  
  # select the last word from the seq_len dimension
  predictions = predictions2[: ,-1:, :]  # (batch_size, 1, vocab_size)
  
  predicted_ids = tf.cast(tf.argsort(predictions,axis = -1, direction='DESCENDING'),tf.int32)[:,:,0:bl]
  predicted_ids = tf.squeeze(predicted_ids)
  predicted_prob = tf.cast(tf.sort(predictions, axis=-1,direction='DESCENDING'), tf.float32)[:,:,0:bl] #[-.7,-.9]
  predicted_prob = tf.squeeze(predicted_prob)
  outputs = tf.stack([tf.concat([output,[[i]]],axis = -1) for i in predicted_ids])
  
 
  for _ in range(MAX_LENGTH):
    temp1,temp2 = [],[]
    for i,j in zip(outputs,predicted_prob):
      combined_mask = create_masks(i)
      predictions1,predictions2, attention_weights = transformer(encoder_input, 
                                                 i,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

      # select the last word from the seq_len dimension
      predictions = predictions2[: ,-1:, :]  # (batch_size, 1, vocab_size)

      
      predicted_ids = tf.cast(tf.argsort(predictions,axis = -1, direction='DESCENDING'),tf.int32)[:,:,0:bl]
      predicted_ids = tf.squeeze(predicted_ids)

      temp1.append(tf.stack([tf.concat([i,[[k]]],axis = -1) for k in predicted_ids]))
      
      
      predicted_prob = tf.cast(tf.sort(predictions, axis=-1,direction='DESCENDING'), tf.float32)[:,:,0:bl] 
      predicted_prob = tf.squeeze(predicted_prob) 
      temp2 += [j+i for i in predicted_prob] 
    
    idx = np.argsort(temp2)
    idx = list(reversed(idx))[0:bl]
    
    t1,t2 = [],[]
    temp1 = tf.concat([i for i in temp1],axis=0)
    
    for i in idx:
      t1.append(temp1[i])
      t2.append(temp2[i])

    
    outputs = tf.stack([i for i in t1])
    predicted_prob = t2
    

  o = tf.squeeze(outputs[0,:,:])
  return o, None




def translate(img,plot=''):
  result, attention_weights = evaluate(img)
  #result_beam1,_ = evaluate_beam(img,2)
  #result_beam2,_ = evaluate_beam(img,3)
  predicted_sentence1 = tokenizer.decode([i for i in result])
  #predicted_sentence2 = tokenizer.decode([i for i in result_beam1])
  #predicted_sentence3 = tokenizer.decode([i for i in result_beam2])

  print('Predicted translation: {}'.format(str(predicted_sentence1)))
  #print('Predicted translation for bl 2: {}'.format(str(predicted_sentence2)))
  #print('Predicted translation for bl 3: {}'.format(str(predicted_sentence3)))
  #return predicted_sentence1,predicted_sentence2,predicted_sentence3
  return predicted_sentence1
  
 



p1 = load_dataset_file(args.test_dir)

videos_test,captions_test,glosses_test = [],[],[]
for i in range(len(p1)):
  videos_test.append(np.asarray(p1[i]['sign']))
  captions_test.append(p1[i]['text'])
  glosses_test.append(p1[i]['gloss'])

"""
if __name__== "__main__":
  tuner = MyTuner(oracle=kt.oracles.BayesianOptimization(
          objective=kt.Objective('epoch_bleu', 'max'),
          max_trials=25),
      hypermodel=build_model,
      directory='finetune',
      project_name='peyman_finetune',
      overwrite = True)


  tuner.search(train_ds=dataset)
  for i in tuner.get_best_hyperparameters(num_trials = 10):
    print(i.values)
  best_hps = tuner.get_best_hyperparameters()[0]
  print(best_hps.values)
  tuner.results_summary()

"""
  #ref,hyp1,hyp2,hyp3,temp = [],[],[],[],[]
  ref,hyp1,temp = [],[],[]
  for m,n in zip(videos_test,captions_test):
    #translated1,translated2,translated3 = translate(m,False)
    translated1 = translate(m,False)
    
    hyp1.append(re.sub(r'^\s+|\s+$','',re.sub(r'\.', '', translated1)))
    #hyp2.append(re.sub(r'^\s+|\s+$','',re.sub(r'\.', '', translated2)))
    #hyp3.append(re.sub(r'^\s+|\s+$','',re.sub(r'\.', '', translated3)))
    temp.append(re.sub(r'^\s+|\s+$','',re.sub(r'\.', '', n)))
    print(n)
    print('\n\n')
  ref= [temp]
    

  #print(hyp)
  #print(ref)
  score_bleu1 = sacrebleu.corpus_bleu(hyp1,ref)
  #score_bleu2 = sacrebleu.corpus_bleu(hyp2,ref)
  #score_bleu3 = sacrebleu.corpus_bleu(hyp3,ref)
  print('bleu greedy is {}'.format(score_bleu1.score))
  #print('bleu bl=1 is {}'.format(score_bleu2.score))
  #print('bleu bl=2 is {}'.format(score_bleu3.score))
"""

 
  





