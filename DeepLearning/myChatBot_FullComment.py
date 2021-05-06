#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 17:01:15 2018

@author: Bryan
"""
# Building a ChatBot with Deep NLP

# Importing the libraries

import numpy as np
import tensorflow as tf
import re 
import os
import time

os.chdir('/home/bryan/Documents/PythonChatbot')


########### PART 1 - DATA PREPROCESSING

# open file and split on the lines
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')


# Creating a disctionary that maps each line to it's id
# looking to create a dataset with two columns, input and output 

id2line = {}
for line in lines:
    
    # create a temporary variable _line
    # complate lines have a length of 5 items in them
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        # get the 5th item from the text file
        id2line[_line[0]] = _line[4]
        
# Creating a list of the conversations
# just need the list of line ids

conversations_ids = []

# exclude the last row
# split and get the last element of the conversation lists
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))
    

# Getting separately the questions and the answers
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        # format is set up so that they are in pairs
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

# Doing the first cleaning of the text
# Convert to lowercase
# expand the contractions first, need to escape out some leading apostrophes
# remove apostrophe's        
        
        
        
        
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~\./,]", "", text)
    return text

# cleaning the questions
# create a new clean list
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
# clean the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# create dictionary that maps each word to its number of occurences
# first get the words in the questions    
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
        
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Create two dictionaries that map the questions words and the answers words to a unique integer
# remove 5% of the least appearing words
# set word threshold to 20
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1

answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1

# Adding the last tokens to these two dictionaries
# add them at the end of the dictionaries        
# had, end of string, out (not in dictionary) filtered out, start of string
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

# add them to each dictionary
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
    
    
    
# Creating the inverse dictionary of the asnwerswords2int dictionary
# creates a diction of the inverses
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}    

    
# Adding the End of String token to the end of every answer
# need the EOS for decoding add a space for padding
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"



# Translating all the questions and answers into integers
# and Replacing all the words that were filtered out by <OUT>
questions_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)
    
    
answers_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)   

# Sorting questions and answers by the length of questions
# speeds up the learning process, start with the shortest first
# at the +1 since the last one is not included in a python range
# get the length of the question to see if it mataches the starting length
# use enumerate to get the index and the question
# Can't sort so need to make a new list
sorted_clean_questions = []
sorted_clean_answers = []

# set a length of 25 for max sentence length
for length in range(1,25 + 1):
    for i in enumerate(questions_to_int):
        # check the length so see if we are at length
        if len(i[1]) == length:
            # sort by the length of questions
            # get the corresponding answer too
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            
            
######  Part 2 - Building the SEQ2SEQ Model  ########

# Creating placeholders for the inputs and targets
            
def model_inputs():
    # 3 inputs, type and dimensions of inputs - list of integers
    # these are 2D matrices
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    # control the dropout rate
    # dropout has been changed to keep probability
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


# Preprocessing the targets selecting a batch size
# Decoder only accepts a certain format of the targets
# Targets must be in batches - targets are sorted_clean_answers
# Each of the answers in the batch of targets must start with <SOS> token in the form of a unique integer for the token



def preprocess_targets(targets, word2int, batch_size):
    # contains a vector of batch size containing the SOS tokens
    # fill the matrix with id' of SOS tokens
    # second argument fills with the integer id of SOS token
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
   
    # use strided_slice to get subset of a tensor
    # second argument is from where to start 0,0
    # then the end, length or batchsize, last column
    # stride is [1,1]
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    # what to concatenate - 1 is Axis
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# CREATE THE RNN ENCODER LAYER
# this is a stacked LSTM with dropout
# defines the architecture
# LSTM vs GRU (Gated Recurrent Unit)
# RNN size is the number of input tensors of encoder layer
# number of layers
# keep prob for dopout
# list of the length of each question in the batch
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    # use basic lstm Cell 
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # apply droput to lstm
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    # encoder cell is composed of several LSTM layers with dropout applied
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    
    # two elements are returned by bidirectionl dynamic function
    # _, is just a placeholder to get us the encoder state
    # builds forward and backward RNN - sizes must match
    # sequence length is the length of the list in each batch
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state



# DECODING THE TRAINING SET
# need the encoder state to proceed to decoding
# decoder embeddings is a mapping from discrete object like words to vectors
# variable scope is advanced dta structure to wrap the variables
# output function is function used to output decoder
def decode_training_set(encoder_state, decoder_cell, 
                        decoder_embedded_input, 
                        sequence_length,decoding_scope,
                        output_function, keep_prob, batch_size):
    
    # initialize attention states as 3D matrices of 0s batch_size x 1 x 
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    
    # use the prepare attention function
    # use the initialized attention state
    # attention keys are keys compared to target state with attention score function
    # attention option
    attention_keys, attention_values,attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                       attention_option = 'bahdanau',
                                       num_units = decoder_cell.output_size)
    
    # from seq2seq module of tf
    # want the attention_decoder_fn_train
    # need all the attention parameters
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = 'attn_dec_train')
    
    # returns decoder_final_state and decoder_final_context_state
    # as unused values marked by _
    
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope = decoding_scope)
    
    
    # apply dropout to decoder
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
                                                                                                            

# DECODING THE TEST/VALIDATION STATE
# decode observations in the test/validation set
# sos = start of string id
# eos = end of string id
# get length of answerswirds2int = num_words
# added arguments needed for attention_decoder_fn_inference function
def decode_test_set(encoder_state, decoder_cell, 
                        decoder_embeddings_matrix, 
                        sos_id, eos_id,
                        max_length,
                        num_words,
                        decoding_scope,
                        output_function, keep_prob, batch_size):
    
    
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,
                                       attention_option = 'bahdanau',
                                       num_units = decoder_cell.output_size)
    # get the additional arguments passed in to the main function
    # decoder embeddings matrix gives
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              max_length,
                                                                              num_words,
                                                                              name = 'attn_dec_inf')
    
    
    # returns decoder_final_state and decoder_final_context_state
    # as unused values marked by _
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,test_decoder_function,scope = decoding_scope)
    return test_predictions
                                                     


# CREATING THE DECODER RNN
def decoder_rnn(decoder_embedded_input,
                decoder_embeddings_matrix,
                encoder_state,
                num_words, 
                sequence_length, 
                rnn_size, 
                num_layers, 
                word2int, 
                keep_prob,
                batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        # create lstm layer
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        # create droppout for 1 lstm layer
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
            
        # apply dropout to the multiple stacked layers
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
            
        # initialize the weights associated to neurons of fully connected layer
        # use the truncated normal distribution
        weights = tf.truncated_normal_initializer(stddev = 0.1)
            
        # create the biases as zeros
        biases = tf.zeros_initializer()
            
        # function returns a fully connected layer as last layer of RNN
        # get the features from previous layers and return scores
        # num_words is number of outputs
        # None for no normalization
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
            
        # use the previously created decode_training_set function
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
            
        # reuse variables introduced in decoding scope
        decoding_scope.reuse_variables()
            
        # use the decode_test_set_function created about to return predictions
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions


# BUILDING THE SEQ2SEQ MODEL
# brain of the chatbot
# inputs = questions
# targets are the answers that we have
# answers/questions num words total number of words
# embedding size is the dimensions of the embedding matrix for the encoder, same for decoder
# rnn size used before
# number of layers with dropout applied
# the dictionary for preprocessing
# function returns the training and test predictions
def seq2seq_model(inputs, 
                  targets, 
                  keep_prob, 
                  batch_size, 
                  sequence_length, 
                  answers_num_words, 
                  questions_num_words,
                  encoder_embedding_size,
                  decoder_embedding_size,
                  rnn_size,
                  num_layers,
                  questionswords2int):
    
    # returns embedded input of encoder
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              # add 1 for the exclusion of upper bound
                                                              answers_num_words + 1,
                                                              
                                                              #number of dimensions in 
                                                              # embedding encoder matrix
                                                              encoder_embedding_size,
                                                              initializer = 
                                                              tf.random_uniform_initializer(0,1))
    # encoder state is output of encoder which is
    # input for the decoder
    encoder_state = encoder_rnn(encoder_embedded_input,
                                rnn_size,
                                num_layers,
                                keep_prob,
                                sequence_length)
    
    # targets preprocessed for back propagation
    preprocessed_targets = preprocess_targets(targets,
                                              questionswords2int,
                                              batch_size)
    
    # dimensions of embedding matrix are the inputs
    # initialize and fill the matrix with random values
    # number of lines x embedding size, lower bound, upper bound of numbers
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1,
                                                               decoder_embedding_size], 0, 1))
    
    
    decoder_embedded_inputs = tf.nn.embedding_lookup(decoder_embeddings_matrix,
                                                     preprocessed_targets)
    
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_inputs,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

############## PART 3 - TRAINING  THE SEQ2SEQ MODEL ############
    
# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining the session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length (originally set to 25)
# None for shape tensor of maximum length
# name for the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'seqence_length')


# Getting the shape of the inputs tensor
# one of the arguments for the trainng function using the ones function
input_shape = tf.shape(inputs)

# Getting the training and test predictions
# reverse is similar to the reshape function in numpy
# specify -1 to perform the reshaping
# encoder embedding size is number of columns in matrix
# same as decoding embedding size
training_predictions, test_predictions  = seq2seq_model(tf.reverse(inputs, [-1]),
                                                                   targets,
                                                                   keep_prob,
                                                                   batch_size,
                                                                   sequence_length,
                                                                   len(answerswords2int),
                                                                   len(questionswords2int),
                                                                   encoding_embedding_size,
                                                                   decoding_embedding_size,
                                                                   rnn_size,
                                                                   num_layers,
                                                                   questionswords2int)




# Setting up the Loss Error, the Optimizer and Gradient Clipping
# gradient clipping avoids exploding or vanishing gradient issue
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    # compute the gradients
    gradients = optimizer.compute_gradients(loss_error)
    
    # set the limits for gradients - 5 to 5
    # loop through all the gradients and clip by value
    # set up as a tuple ((clip by value, grad variable))
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor,
                         grad_variable in gradients if grad_tensor is not None]
    
    # apply graient clipping to optimizer
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# apply the padding to the sequences with pad token - all must have the same length
# Question: ['Who', 'are', 'you']  ==> ['Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>]
# Answer: ['I', 'am', 'a', 'bot', '.'] ==> [<SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>, <PAD>]
def apply_padding(batch_of_sequences, word2int):
    
    # find the length of the longest sequence in the batch
    # for loop in a list for list comprehension
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    
    # use max sequence length and subtract sequence to get needed tokens
    # add two lists sequence + padid * neeeded extra tokens
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
# use the // operator to get an integer
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        # start with the questions
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        
        # answers
        answers_in_batch = answers[start_index : start_index + batch_size]
        
        # ensure padded with apply padding function
        # need np.arrays for tensorflow
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        
        # use yield when dealing with sequences
        yield padded_questions_in_batch, padded_answers_in_batch


# split the data Q's and A's into training and validation use a 85/15 split
training_validation_split = int(len(sorted_clean_questions) * 0.15)

# training Q & A
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]

# testing Q & A
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions))//batch_size//2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = 'chatbot_weights.ckpt'
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error],
                                                   {inputs: padded_questions_in_batch,
                                                    targets: padded_answers_in_batch,
                                                    lr: learning_rate,
                                                    sequence_length: padded_answers_in_batch.shape[1],
                                                    keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print("Epoch {:>3}/{}, Batch: {:4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds".format(epoch,
                  epochs,
                  batch_index,
                  len(training_questions)//batch_size,
                  total_training_loss_error/batch_index_check_training_loss,
                  int(batch_time * batch_index_check_training_loss)))
            total_taining_loss_error = 0
            
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions,validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error,
                                                   {inputs: padded_questions_in_batch,
                                                    targets: padded_answers_in_batch,
                                                    lr: learning_rate,
                                                    sequence_length: padded_answers_in_batch.shape[1],
                                                    keep_prob: 1})
            total_validation_loss_error += batch_validation_loss_error
    ending_time = time.time()
    batch_time = ending_time - starting_time
    average_validation_loss_error = total_validation_loss_error / (len(validation_questions)/batch_size)
    print('Validation Loss Error: {:>6.3f}, Batch Balidation Time: {:d} seconds'.format(average_validation_loss_error,
          int(batch_time)))
    learning_rate *= learning_rate_decay
    if learning_rate < min_learning_rate:
        learning_rate = min_learning_rate
    list_validation_loss_error.append(average_validation_loss_error)
    if average_validation_loss_error < min(list_validation_loss_error):
        print('I speak better now!!')
        early_stopping_check = 0
        saver = tf.train.Saver()
        saver.save(session, checkpoint)
    else:
        print("Sorry, I do not speak better, I need to practive more")
        early_stopping_check +=1
        if early_stopping_check == early_stopping_stop:
            break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore.  This is the best I can do.")
        break
print("Game over")
        












