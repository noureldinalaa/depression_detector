#imports
from parsing_data import Depression_detection
from data_preprocessing import preprocessing
from training_testing import training_testing
from training_testing import SentimentRNN
from pathlib import Path
import pandas as pd
import pickle

import torch
import torch.nn as nn


from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from textwrap import wrap

from Bert import Data_loader,DataSetDepression,BERTClassifier,training_Bert,testing_Bert
from sklearn.utils import shuffle

import bcolz

# get th parent path
base_path = Path.cwd().parent #Nour
# base_path = Path.cwd() #Adrian
training_positive_path = base_path.joinpath('./2017/train/positive_examples_anonymous_chunks')
training_negative_path = base_path.joinpath('./2017/train/negative_examples_anonymous_chunks')
test_path = base_path.joinpath('./2017/test')

Dd = Depression_detection(base_path,
                          training_positive_path,
                          training_negative_path,
                          test_path)

Dp_training = preprocessing()
Dp_testing = preprocessing()

RNN_preparation =  training_testing()


## Concatenate all the frames for each folder after parsing them
training_positive_dateframe = pd.concat(Dd.parse_folder(training_positive_path))
training_negative_dataframe = pd.concat(Dd.parse_folder(training_negative_path))
test_dataframe = pd.concat(Dd.parse_folder(test_path))

## add labels to positive and negative subjects training dataset
training_positive_dateframe['LABEL'] = 1
training_negative_dataframe['LABEL'] = 0

## adding label to test dataframe
test_dataframe = Dd.prepare_test_dataframe(test_dataframe)

## save them to csv file
training_positive_dateframe.to_csv('training_positive_dateframe.csv')
training_negative_dataframe.to_csv('training_negative_dataframe.csv')
test_dataframe.to_csv('test_dataframe.csv')



positive_training_file_CSV = 'training_positive_dateframe.csv'
negative_training_file_CSV = 'training_negative_dataframe.csv'



unified_training_df = Dd.Unifing_training_data(positive_training_file_CSV,negative_training_file_CSV)
print(unified_training_df.shape)
unified_training_df.set_index('ID')

## Concatentenate title with text

unified_training_df["TITLE_TEXT"] = unified_training_df["TITLE"] + unified_training_df["TEXT"]
unified_training_df.to_csv('unified_training_df.csv')
unified_training_df = pd.read_csv('unified_training_df.csv')
unified_training_df = unified_training_df.drop(['TITLE', 'INFO', 'TEXT' ], axis=1)
#unify test dataframe
test_df = pd.read_csv('test_dataframe.csv')
test_df.set_index('ID')
test_df["TITLE_TEXT"] = test_df["TITLE"] + test_df["TEXT"]
test_df.to_csv('unified_test_df.csv')
unified_test_df = pd.read_csv('unified_test_df.csv')
unified_test_df = unified_test_df.drop(['TITLE', 'INFO', 'TEXT' ], axis=1)


tokens = Dp_training.tokenization(unified_training_df,train=True)
test_tokens = Dp_testing.tokenization(unified_test_df)


with open("tokens.pickle","rb") as file:
    pickle_output = pickle.load(file)
tokens  = pickle_output

with open("tokens_test.pickle","rb") as file:
    pickle_output = pickle.load(file)
tokens_test = pickle_output

## Convert tokens to integer

vocab_to_ints_training = Dp_training.vocab_to_int(tokens)

## Convert tokens to integer(for Test)
# vocab_to_ints_testing = Dp_testing.vocab_to_int(tokens_test)

## Preprocessing dataframe

# Dp_training.dataframe_preprocessing(unified_training_df,train=True)
unified_training_df_preprocessed = pd.read_csv('./unified_training_df_preprocessed.csv')

#Preprocessing for testing
# Dp_testing.dataframe_preprocessing(unified_test_df)
unified_testing_df_preprocessed = pd.read_csv('./unified_testing_df_preprocessed.csv')

# we have unbalanced data in which non depressed data is much more
# than depressed data,and the model will tend to predict
# undepressed if it isn't downsampled .
Dp_training.downsampling(unified_training_df_preprocessed)
downsampled_data = pd.read_csv('./downsampled_data.csv')
downsampled_data= shuffle(downsampled_data, random_state=0)

#get training_text_ints
text_integers_training = Dp_training.text_ints_extract(downsampled_data)

#get testing_text_ints
text_integers_testing = Dp_testing.text_ints_extract(unified_testing_df_preprocessed )


#pad the features
#get the labels
#convert to numpy
seq_length = 100

#training Dataset
training_features = Dp_training.pad_features(text_integers_training, seq_length)
training_labels = Dp_training.get_labels(downsampled_data.LABEL)


#testing Dataset

testing_features = Dp_testing.pad_features(text_integers_testing, seq_length)
testing_labels = Dp_testing.get_labels(unified_testing_df_preprocessed.LABEL)



#
# #  ------- >Bert
#
# RANDOM_SEED_BERT = 42
# MAX_LEN_BERT = 10        #Max length training:  2400  , Max length testing:  7390
# BATCH_SIZE_BERT = 2
# CLASSES_BERT = 2
#
# np.random.seed(RANDOM_SEED_BERT)
# torch.manual_seed(RANDOM_SEED_BERT)
# PRE_TRAINED_MODEL_NAME = "bert-base-cased"
# tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#
# device_for_BERT = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# #preparing_training_data
# df_train =downsampled_data
# df_train = df_train.drop(['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1.1','ID', 'DATE',
#        'urls_out', 'punctuation_out','text_tokens', 'text_ints', 'length'], axis=1)
#
# #preparing_test_data
# df_test = unified_testing_df_preprocessed
# df_test = df_test.drop(['Unnamed: 0', 'Unnamed: 0.1', 'ID', 'TITLE', 'DATE', 'INFO', 'TEXT',
#         'urls_out', 'punctuation_out', 'text_tokens',
#        'text_ints', 'length'], axis=1)
#
#
#
# data_loader_bert =Data_loader()
#
# #split train data to train and validation
# df_train,df_validation = train_test_split(df_train,test_size = 0.2 , random_state =RANDOM_SEED_BERT)
#
#
# train_data_loader = data_loader_bert.data_loader(df_train , tokenizer, MAX_LEN_BERT, BATCH_SIZE_BERT)
# validation_data_loader = data_loader_bert.data_loader(df_validation , tokenizer, MAX_LEN_BERT, BATCH_SIZE_BERT)
# test_data_loader = data_loader_bert.data_loader(df_test, tokenizer, MAX_LEN_BERT, BATCH_SIZE_BERT)
#
#
# #call Bert model
# model_BERT = BERTClassifier(CLASSES_BERT,PRE_TRAINED_MODEL_NAME = "bert-base-cased")
# model_BERT = model_BERT.to(device_for_BERT)
#
# EPOCHS = 1
# optimizer = AdamW(model_BERT.parameters(), lr=2e-5, correct_bias=False)
# total_steps = len(train_data_loader) * EPOCHS
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=total_steps
#
# )
# loss_fn = nn.CrossEntropyLoss().to(device_for_BERT)
#
#
# #Training BERT
# training_Bert =training_Bert()
# #training_Bert.train_model_preparation(model_BERT, train_data_loader, loss_fn, optimizer, device_for_BERT, scheduler, len(df_train))
# training_Bert.train_model(model_BERT,train_data_loader,validation_data_loader,loss_fn,optimizer,device_for_BERT,scheduler,df_train,df_validation,EPOCHS)
#
#
#
# model_BERT.load_state_dict(torch.load('model_trained_Bert_pretrained.pt'))
#
# #testing BERT
# testing_Bert = testing_Bert()
# # testing_Bert.eval_model_preparation(model_BERT, validation_data_loader, loss_fn, device_for_BERT, len(df_test))
# testing_Bert.test_model(model_BERT,validation_data_loader,loss_fn,device_for_BERT,df_test)



# -------- >RNN

#training_testing_phase
#split the data anch convert it from numpy to torch for RNN
split_frac = 0.8
batch_size = 50


train_on_gpu = RNN_preparation.gpu_check()

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_ints_training)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 512 #128
n_layers = 3 #2

RNN_net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers,weights_matrix=0,train_on_gpu=train_on_gpu)

print(RNN_net)
#
lr = 0.0001 #0.01
epochs = 5  #10
#
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(RNN_net.parameters(), lr=lr)



for i in range(5):
    train_loader,valid_loader,test_loader = RNN_preparation.loader_creation(training_features,training_labels,testing_features
                                   ,testing_labels,split_frac,batch_size,idx=(i+1))

    RNN_preparation.RNN_training(RNN_net,lr=lr,epochs = epochs,train_on_gpu =train_on_gpu
                        ,batch_size=batch_size,train_loader=train_loader,valid_loader=valid_loader,criterion =criterion ,optimizer=optimizer)

    RNN_net.load_state_dict(torch.load('model_trained_RNN_not_pretrained2.pt'))



    RNN_preparation.RNN_test(RNN_net,lr=lr,epochs = epochs,train_on_gpu =train_on_gpu
                        ,batch_size=batch_size, test_loader=test_loader,criterion =criterion,optimizer=optimizer )




# ------ >training with twitter golve pretrained model



# words = []
# idx = 0
# word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir='glove.twitter.27B.200d.dat', mode='w')
#
# with open('glove.twitter.27B.200d.txt', 'rb') as f:
#     for l in f:
#         line = l.decode().split()
#         word = line[0]
#         words.append(word)
#         word2idx[word] = idx
#         idx += 1
#         vect = np.array(line[1:]).astype(np.float)
#         vectors.append(vect)
#
# #1.2 million vocab in twitter Glove
# pickle.dump(words, open('glove.twitter.27B.200d_words.pkl', 'wb'))
# pickle.dump(word2idx, open('glove.twitter.27B.200d_idx.pkl', 'wb'))
#
# vectors = bcolz.carray(vectors[:].reshape((1193514, 200)), rootdir='glove.twitter.27B.200d.dat', mode='w')
# # vectors.flush()


# vectors = bcolz.open('glove.twitter.27B.200d.dat')[:]
# words = pickle.load(open('glove.twitter.27B.200d_words.pkl', 'rb'))
# word2idx = pickle.load(open('glove.twitter.27B.200d_idx.pkl', 'rb'))
#
# glove = {w: vectors[word2idx[w]] for w in words}
#
# matrix_len = vocab_size
# weights_matrix = np.zeros((matrix_len, 200))
# words_found = 0
#
# for i, word in enumerate(vocab_to_ints_training.keys()):
#     try:
#         weights_matrix[i] = glove[word]
#         words_found += 1
#     except KeyError:
#         weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))
#
#
# pickle.dump(weights_matrix, open('weights_matrix_glove.twitter.27B.200d_words.pkl', 'wb'))

# weights_matrix = pickle.load(open('weights_matrix_glove.twitter.27B.200d_words.pkl', 'rb'))
#
# weights_matrix = torch.from_numpy(weights_matrix)
# weights_matrix  = weights_matrix.cuda()
# RNN_net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers,weights_matrix,train_on_gpu=train_on_gpu,pretrained=True)
# #
# print(RNN_net)
#
# lr = 0.0001 #0.01
# epochs = 5 #10
#
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(RNN_net.parameters(), lr=lr)

# #

# #
# #
# # # Not to be used as it is used with his line self.embedding = nn.Embedding.from_pretrained(weights_matrix)
# # RNN_net.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
# # RNN_net.embedding.weight.requires_grad = True
#
# train_loader,valid_loader,test_loader = RNN_preparation.loader_creation(training_features,training_labels,testing_features
#                                ,testing_labels,split_frac,batch_size,idx=2)
# #
# RNN_preparation.RNN_training(RNN_net,lr=lr,epochs = epochs,train_on_gpu =train_on_gpu
#                     ,batch_size=batch_size,train_loader=train_loader,valid_loader=valid_loader,criterion =criterion ,optimizer=optimizer)
#
# RNN_net.load_state_dict(torch.load('model_trained_RNN_pretrained.pt'))
#
# RNN_preparation.RNN_test(RNN_net,lr=lr,epochs = epochs,train_on_gpu =train_on_gpu
#                     ,batch_size=batch_size, test_loader=test_loader,criterion =criterion,optimizer=optimizer )


