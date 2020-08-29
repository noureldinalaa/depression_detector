#imports
from parsing_data import Depression_detection
from data_preprocessing import preprocessing
from pathlib import Path
import pandas as pd
import pickle

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

## Concatenate all the frames for each folder after parsing them
# training_positive_dateframe = pd.concat(Dd.parse_folder(training_positive_path))
# training_negative_dataframe = pd.concat(Dd.parse_folder(training_negative_path))
# test_dataframe = pd.concat(Dd.parse_folder(test_path))

## add labels to positive and negative subjects training dataset
# training_positive_dateframe['LABEL'] = 1
# training_negative_dataframe['LABEL'] = 0

## adding label to test dataframe
# test_dataframe = Dd.prepare_test_dataframe(test_dataframe)

## save them to csv file
# training_positive_dateframe.to_csv('training_positive_dateframe.csv')
# training_negative_dataframe.to_csv('training_negative_dataframe.csv')
# test_dataframe.to_csv('test_dataframe.csv')



# positive_training_file_CSV = 'training_positive_dateframe.csv'
# negative_training_file_CSV = 'training_negative_dataframe.csv'



# unified_training_df = Dd.Unifing_training_data(positive_training_file_CSV,negative_training_file_CSV)
# print(unified_training_df.shape)
# unified_training_df.set_index('ID')

## Concatentenate title with text

# unified_training_df["TITLE_TEXT"] = unified_training_df["TITLE"] + unified_training_df["TEXT"]
# unified_training_df.to_csv('unified_training_df.csv')
unified_training_df = pd.read_csv('unified_training_df.csv')
unified_training_df = unified_training_df.drop(['TITLE', 'INFO', 'TEXT' ], axis=1)
#unify test dataframe
test_df = pd.read_csv('test_dataframe.csv')
test_df.set_index('ID')
test_df["TITLE_TEXT"] = test_df["TITLE"] + test_df["TEXT"]
test_df.to_csv('unified_test_df.csv')
unified_test_df = pd.read_csv('unified_test_df.csv')


# tokens = Dp_training.tokenization(unified_training_df,train=True)
# test_tokens = Dp_testing.tokenization(unified_test_df)


with open("tokens.pickle","rb") as file:
    pickle_output = pickle.load(file)
tokens  = pickle_output

with open("tokens_test.pickle","rb") as file:
    pickle_output = pickle.load(file)
tokens_test = pickle_output

## Convert tokens to integer

vocab_to_ints_training = Dp_training.vocab_to_int(tokens)

## Convert tokens to integer(for Test)
vocab_to_ints_testing = Dp_testing.vocab_to_int(tokens_test)

## Preprocessing dataframe

# Dp_training.dataframe_preprocessing(unified_training_df,train=True)
unified_training_df_preprocessed = pd.read_csv('./unified_training_df_preprocessed.csv')

#Preprocessing for testing
Dp_testing.dataframe_preprocessing(unified_test_df)
unified_testing_df_preprocessed = pd.read_csv('./unified_testing_df_preprocessed.csv')

# we have unbalanced data in which non depressed data is much more
# than depressed data,and the model will tend to predict
# undepressed if it isn't downsampled .
Dp_training.downsampling(unified_training_df_preprocessed)
downsampled_data = pd.read_csv('./downsampled_data.csv')

#get training_text_ints
text_integers_training = Dp_training.text_ints_extract(downsampled_data)

#get testing_text_ints
text_integers_testing = Dp_testing.text_ints_extract(unified_testing_df_preprocessed )


#pad the features
#get the labels
#convert to numpy
seq_length = 50

#training Dataset
training_features = Dp_training.pad_features(text_integers_training, seq_length)
training_labels = Dp_training.get_labels(downsampled_data.LABEL)


#testing Dataset

testing_features = Dp_testing.pad_features(text_integers_testing, seq_length)
testing_labels = Dp_testing.get_labels(unified_testing_df_preprocessed.LABEL)