# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import sys

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_func as tf_func

import pickle

user_song_data = {}
# Read the data into a list of strings.
def read_data(filename):
  #Extract the first file enclosed in a zip file as a list of words
  data = open(filename,'r').readlines()
  words = []
  songs = []
  for user in data:
    user = user.strip('\n')
    print (user)
    user = user.split(',')
    user = user[:-1]
    print(user)
    if user[0] in user_song_data:
      user_song_data[user[0]].append(user[1:len(user) ])
    else:
      user_song_data[user[0]] = []
      user_song_data[user[0]].append(user[1:len(user) ])
    words.extend(user[2:len(user)])
    songs.append(user[1])

  return words,songs
words,songs = read_data('data.txt')
vocabulary_size = len(set(words)) + 1
song_size = len(set(songs))

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count += 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data_words, count_words, dictionary_words, reverse_dictionary_words = build_dataset(words)
data_songs, count_songs, dictionary_songs, reverse_dictionary_songs = build_dataset(songs)

# print (data_words)
# print ('count')
# print (count_words)
# print ('dictionary')
print (dictionary_words)
# print ('reverse_dictionary')
# print (reverse_dictionary_words)

# print ('*****')

# print (data_songs)
# print ('count')
# print (count_songs)
# print ('dictionary')
# print (dictionary_songs)
# print ('reverse_dictionary')
# print (reverse_dictionary_songs)

data_index = 0
song_data = {}

def generate_pairs(user_songs):
    outer = []
    inner = []
    #print (user_songs)
    for i in range(len(user_songs)):
      words_inner = [0]*(vocabulary_size)
      for word in user_songs[i][1:len(user_songs[i])]:
        words_inner[dictionary_words[word]] = dictionary_words[word]
      song_data[user_songs[i][0]] = words_inner
      for j in range(len(user_songs)):
        if i != j:
          words_outer = [0]*(vocabulary_size)
          for word in user_songs[j][1:len(user_songs[j])]:
            words_outer[dictionary_words[word]] = dictionary_words[word] 
          outer.append(words_outer)
          inner.append(words_inner)
    #print (outer)
    return outer,inner

def generate_pairs_last(user_songs):
    outer = []
    inner = []
    #print (user_songs)
    j = len(user_songs) - 1
    for i in range(len(user_songs)):
      words_inner = [0]*(vocabulary_size)
      for word in user_songs[i][1:len(user_songs[i])]:
        words_inner[dictionary_words[word]] = dictionary_words[word]
      song_data[user_songs[i][0]] = words_inner
      if i != j:
          words_outer = [0]*(vocabulary_size)
          for word in user_songs[j][1:len(user_songs[j])]:
            words_outer[dictionary_words[word]] = dictionary_words[word] 
          outer.append(words_outer)
          inner.append(words_inner)
    #print (outer)
    return outer,inner

def generate_pairs_half(user_songs):
    outer = []
    inner = []
    #print (user_songs)
    for i in range(len(user_songs)/2):
      words_inner = []
      for word in user_songs[i][1:len(user_songs[i])]:
        words_inner.append(dictionary_words[word])
      song_data[user_songs[i][0]] = words_inner
      for j in range(len(user_songs)/2 : len(user_songs)):
        if i != j:
          words_outer = []
          for word in user_songs[j][1:len(user_songs[j])]:
            words_outer.append(dictionary_words[word])
          outer.append(words_outer)
          inner.append(words_inner)
    #print (outer)
    return outer,inner


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_data():
    # global data_index
    # assert batch_size % num_skips == 0
    # assert num_skips <= 2 * skip_window
    # outer = np.ndarray(shape=(batch_size), dtype=np.int32)
    # inner = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    outer = []
    inner = []
    for user in user_song_data:
      sub_outer,sub_inner = generate_pairs(user_song_data[user])
      outer.extend(sub_outer)
      inner.extend(sub_inner)
    return outer, inner

# Step 4: Build and train a skip-gram model.

trainFeats, trainLabels = generate_data()
#pickle.dump([trainFeats, trainLabels], open('generate_data.model', 'wb'))
batch_size = 100
word_embedding_size = 128 # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 4         # How many times to reuse an input to generate a label.
num_sampled = 64
num_features = 28

graph = tf.Graph()
embedding_song = np.zeros((song_size,vocabulary_size*word_embedding_size))
embedding_array = np.zeros((vocabulary_size,word_embedding_size))
#embedding_array[0] =  np.zeros(word_embedding_size)
for i in range(0,len(embedding_array)):
      embedding_array[i] = np.random.rand(word_embedding_size)*0.02-0.01

with graph.as_default():

  # Input data.
  print("Entered the graph")
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size,num_features])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size,num_features])
  #valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, word_embedding_size], -1.0, 1.0))
    #embeddings = tf.constant(embedding_array, dtype=tf.float32)
    # embeddings = tf.Variable(
    #     tf.truncated_normal([vocabulary_size,word_embedding_size],
    #                         stddev=1.0 / math.sqrt(word_embedding_size)))
    embed_outer = tf.nn.embedding_lookup(embeddings, train_inputs)
    embed_outer = tf.reshape(embed_outer,[batch_size,num_features*word_embedding_size])
    # embed_test = tf.nn.embedding_lookup(embeddings, test_inputs)
    # embed_test = tf.reshape(embed_test,[1,Config.n_Tokens*Config.embedding_size])

    weights = tf.Variable(
        tf.truncated_normal([vocabulary_size,word_embedding_size],
                            stddev=1.0 / math.sqrt(word_embedding_size)))
    #biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Get target embeddings from lables - for Cross Entropy loss
    embed_inner = tf.nn.embedding_lookup(weights, train_labels)
    embed_inner = tf.reshape(embed_inner,[batch_size,num_features*word_embedding_size])
    # true_w = tf.nn.embedding_lookup(weights, train_labels)
    # true_w = tf.reshape(true_w, [-1, embedding_size])

    # Sample negative examples with unigram probability - for NCE loss
    #sample = np.random.choice(vocabulary_size, num_sampled, p=unigram_prob, replace=False)

  """
  ===========================================================


  You can change the function 'cross_entropy_loss'
  to 'nce_loss' once you are done with your implementation


  loss = tf.reduce_mean(tf_func.cross_entropy_loss(embed, true_w))
  loss = tf.reduce_mean(tf_func.nce_loss(embed, weights, biases, train_labels, sample)

  ===========================================================
  """
  # #a = tf.nn.nce_loss(weights=weights,
  #                    biases=biases,
  #                    labels=train_labels,
  #                    inputs=embed_outer,
  #                    num_sampled=num_sampled,
  #                    num_classes=song_size)
    
  
  #a = tf.nn.softmax_cross_entropy_with_logits(labels=embed_inner, logits=embed_outer)
  #loss = tf.reduce_mean(tf_func.nce_loss(embed, weights, biases, train_labels, sample))
  loss = tf.reduce_mean(tf_func.cross_entropy_loss(embed_outer, embed_inner))
  # Construct the SGD optimizer using a learning rate of 1.0.
  #loss = tf.reduce_mean(a)
  optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  
  
  # Add variable initializer.
  init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 10000


with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")

  average_loss = 0

  print('Data created')
    
  #trainFeats, trainLabels = pickle.load(open('generate_data.model', 'rb'))
  # print (len(trainFeats))
  # print (len(trainLabels))
  for step in xrange(num_steps):
    start = (step*batch_size)%len(trainFeats)
    end = ((step+1)*batch_size)%len(trainFeats)
    if end < start:
      start -= end
      end = len(trainFeats)
      continue
    batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]
    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    #feed_dict = {train_inputs: trainFeats, train_labels: trainLabels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val= session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 50 == 0:
      if step > 0:
        average_loss /= 50
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0

    # Note that thi9+-+-s is expensive (~20% slowdown if computed every 500 steps)
    # if step % 10000 == 0:
    #   sim = similarity.eval()
    #   for i in xrange(valid_size):
    #     valid_word = reverse_dictionary[valid_examples[i]]
    #     top_k = 8  # number of nearest neighbors
    #     nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    #     log_str = "Nearest to %s:" % valid_word
    #     for k in xrange(top_k):
    #       close_word = reverse_dictionary[nearest[k]]
    #       log_str = "%s %s," % (log_str, close_word)
    #     print(log_str)
  final_embeddings = normalized_embeddings.eval()
  print (len(final_embeddings))

  pickle.dump([user_song_data, song_data, final_embeddings], open(sys.argv[1], 'wb'))

