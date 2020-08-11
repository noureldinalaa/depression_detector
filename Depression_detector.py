#imports
from Models import Depression_detection
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

Dp = preprocessing()

# Concatenate all the frames for each folder after parsing them
training_positive_dateframe = pd.concat(Dd.parse_folder(training_positive_path))
training_negative_dataframe = pd.concat(Dd.parse_folder(training_negative_path))
test_dataframe = pd.concat(Dd.parse_folder(test_path))

#add labels to positive and negative subjects training dataset
training_positive_dateframe['LABEL'] = 1
training_negative_dataframe['LABEL'] = 0

#adding label to test dataframe
test_dataframe = Dd.prepare_test_dataframe(test_dataframe)

#save them to csv file
training_positive_dateframe.to_csv('training_positive_dateframe.csv')
training_negative_dataframe.to_csv('training_negative_dataframe.csv')
test_dataframe.to_csv('test_dataframe.csv')



positive_training_file_CSV = 'training_positive_dateframe.csv'
negative_training_file_CSV = 'training_negative_dataframe.csv'



unified_training_df = Dd.Unifing_training_data(positive_training_file_CSV,negative_training_file_CSV)
print(unified_training_df.shape)
unified_training_df.set_index('ID')

# Concatentenate title with text

unified_training_df["TITLE_TEXT"] = unified_training_df["TITLE"] + unified_training_df["TEXT"]
unified_training_df.to_csv('unified_training_df.csv')
unified_training_df = pd.read_csv('unified_training_df.csv')
unified_training_df = unified_training_df.drop(['TITLE', 'INFO', 'TEXT' ], axis=1)


tokens = Dp.tokenization(unified_training_df)

# with open("tokens.pickle","rb") as file:
#     pickle_output = pickle.load(file)
# tokens  = pickle_output

# Convert tokens to integer

vocab_to_ints = Dp.vocab_to_int(tokens)

# Preprocessing dataframe 

Dp.dataframe_preprocessing(unified_training_df)
unified_training_df_preprocessed = pd.read_csv('./unified_training_df_preprocessed.csv')


