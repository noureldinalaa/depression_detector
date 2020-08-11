import re
import string
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import Counter

import pickle

from sklearn.utils import resample
import pandas as pd

class preprocessing(object):
    def __init__(self):

        nlp = English()
        self.tokenizer = Tokenizer(nlp.vocab)

    def tokenization(self,unified_training_df):
        '''
        Preprocess the whole text ie.
         * remove https links and punctuation
         * remove spaces and numbers
         * lowering the case
        and finally tokenize it.

        :param unified_training_df:
        :return: tokens
        '''

        #get punctautions from string
        punctuations = string.punctuation

        corpus = unified_training_df.TITLE_TEXT
        corpus = corpus.to_list()

        #Take each line of the subject and combine all of them in one big text(corpus)
        text = []
        for x in corpus:
            text.append(x)

        text_joined = ' '
        text = text_joined.join(text)

        #remove hhtps links and punctautions
        text = re.sub(r'http\S+', '', text)
        all_text = ''.join([c for c in text if c not in punctuations])

        #Tokanize the whole text using spacy tokanizer
        words = self.tokenizer(all_text)

        #delete space and numbers and lower case all the tokens
        tokens = [token.lower_ for token in words if not token.is_space and not token.like_num]

        #save it to a pickle file
        saveObject = (tokens )
        with open("tokens.pickle","wb") as file :
            pickle.dump(saveObject,file)

        return tokens

    def vocab_to_int(self,tokens):

        '''
        Annotate all the vocabs in the text with a corresponding integer.
        :return: self.vocab_to_int
        '''

        ## Build a dictionary that maps words to integers
        counts = Counter(tokens)
        vocab = sorted(counts, key=counts.get, reverse=True)
        # print(vocab[:30])
        # 1 in enamurate means the dictionary will begin with one instead of 0
        self.vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

        return self.vocab_to_int
    
    # the next functions is for preprocessing data frame
    def out_urls(self,text, **kwargs):
        '''
        remove urls from the text
        :param text:
        :param kwargs:
        :return: text
        '''
        text = re.sub(r'http\S+', '', text)
        return text

    def out_punctuation(self,text, **kwargs):
        '''
        remove punctuations
        :param text:
        :param kwargs:
        :return: text
        '''
        punctuations = string.punctuation
        text = ''.join([c for c in text if c not in punctuations])
        return text

    def text_token(self,text, **kwargs):
        '''
        tokenize each cell
        :param text:
        :param kwargs:
        :return: tokens
        '''
        words = self.tokenizer(text)
        tokens = [token.lower_ for token in words if not token.is_space and not token.like_num]
        return tokens

    def text_ints(self,tokens, **kwargs):
        '''
        convert each text to integer
        :param tokens:
        :param kwargs:
        :return: text_num
        '''
        text_num = [self.vocab_to_int[word] for word in tokens]
        return text_num

    def lenght_text(self,text_insts, **kwargs):
        '''
        get the length of each text in each cell
        :param text_insts:
        :param kwargs:
        :return: length
        '''

        length = len(text_insts)
        return length

    def dataframe_preprocessing(self,unified_training_df):
        '''
        call the functions of preprocessing
        :param unified_training_df:
        :return:
        '''
        unified_training_df['urls_out'] = unified_training_df['TITLE_TEXT'].apply(self.out_urls, axis='columns')
        unified_training_df['punctuation_out'] = unified_training_df['urls_out'].apply(self.out_punctuation, axis='columns')
        unified_training_df['text_tokens'] = unified_training_df['punctuation_out'].apply(self.text_token, axis='columns')
        unified_training_df['text_ints'] =  unified_training_df['text_tokens'].apply(self.text_ints, axis='columns')
        unified_training_df['length'] = unified_training_df['text_ints'].apply(self.lenght_text, axis='columns')
        # neglect all cells(subjects) with zero length (text and the label,..)
        unified_training_df = unified_training_df[unified_training_df.length != 0]
        unified_training_df.to_csv('./unified_training_df_preprocessed.csv')


    def downsampling(self,unified_training_df_preprocessed):
        '''
        downsampling majority(un)
        :param unified_training_df_preprocessed:
        :return:
        '''
        # separate minority and majority classes
        un_depressed = unified_training_df_preprocessed[unified_training_df_preprocessed.LABEL == 0]
        depressed = unified_training_df_preprocessed[unified_training_df_preprocessed.LABEL == 1]

        undepressed_downsampled = resample(un_depressed,
                                           replace=False,  # sample without replacement
                                           n_samples=len(depressed),  # match minority n
                                           random_state=27)  # reproducible results

        # combine minority and downsampled majority
        downsampled = pd.concat([undepressed_downsampled, depressed])

        # checking counts
        downsampled.LABEL.value_counts()

        #save it to csv file
        downsampled.to_csv('./downsampled_data.csv')

















