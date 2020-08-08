import re
import string
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import Counter

import pickle

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
        :return:
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
        :return:
        '''

        ## Build a dictionary that maps words to integers
        counts = Counter(tokens)
        vocab = sorted(counts, key=counts.get, reverse=True)
        # print(vocab[:30])
        # 1 in enamurate means the dictionary will begin with one instead of 0
        vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

        return vocab_to_int










