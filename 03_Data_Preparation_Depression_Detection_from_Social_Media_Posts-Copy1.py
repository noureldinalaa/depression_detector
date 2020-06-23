#!/usr/bin/env python
# coding: utf-8

# # Depression Detection from Social Media Posts

# Nour Eldin Alaa
# 
# Adrian Granados
# 

# ## Data Preprocessing

# In[1]:


# Libraries

import pandas as pd
from sklearn.utils import resample
import spacy
import string
nlp = spacy.load('en_core_web_lg')
punctuations = string.punctuation
english_stopwords_spacy = spacy.lang.en.stop_words.STOP_WORDS


# In[2]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


spacy.explain('WORK_OF_ART')


# ## Loading  training files

# In[3]:


p_total_df = pd.read_csv('p_total_df.csv', index_col=0)


# In[4]:


p_total_df.set_index('ID')


# In[5]:


p_total_df["TITLE_TEXT"] = p_total_df["TITLE"] + p_total_df["TEXT"]


# In[6]:


#p_total_df


# In[7]:


# p_total_df.to_csv('./p_total_title_text.csv')


# In[8]:


# this function concatenate all text together for each subject
#def subject_aggregation():
column_aggregation = {}

for col in p_total_df.columns:
    column_aggregation[col] = 'first'

column_aggregation["TITLE_TEXT"] = lambda col: ' '.join(map(str, col))

p_unified_df = p_total_df.groupby('ID').agg(column_aggregation)
    #return p_unified_df


# In[9]:


p_unified_df.drop(['ID', 'TITLE', 'INFO', 'TEXT' ], axis=1)
#subject_aggregation(p_total_df)


# In[ ]:


#p_column_aggregation = {}

#for col in p_total_df.columns:
 #   column_aggregation[col] = 'first'

#p_column_aggregation["TEXT"] = lambda col: ' '.join(map(str, col))

#p_unified_df = p_total_df.groupby('ID').agg(p_column_aggregation)


# ## Loading  training files

# In[10]:


n_total_df = pd.read_csv('n_total_df.csv', index_col=0)


# In[12]:


n_total_df.set_index('ID')


# In[13]:


n_total_df["TITLE_TEXT"] = n_total_df["TITLE"] + n_total_df["TEXT"]


# In[14]:


#n_total_df


# In[15]:


#n_total_df.to_csv('./n_total_title_text.csv')


# In[16]:


column_aggregation = {}

for col in n_total_df.columns:
    column_aggregation[col] = 'first'

column_aggregation["TITLE_TEXT"] = lambda col: ' '.join(map(str, col))

n_unified_df = n_total_df.groupby('ID').agg(column_aggregation)


# In[17]:


# n_unified_df


# In[19]:


n_unified_df.drop(['ID', 'TITLE', 'INFO', 'TEXT' ], axis=1)


# In[20]:


unified_df = pd.concat([p_unified_df, n_unified_df]) 


# In[22]:


unified_df = unified_df.drop(['ID', 'TITLE', 'INFO', 'TEXT' ], axis=1)


# In[25]:

# unified_df


# In[24]:


#unified_df.to_csv('./unified_training.csv')


# In[29]:


#X = unified_df.reset_index().drop(["ID", 'DATE'], axis=1)


# In[30]:





# y = df.Class
# X = df.drop('Class', axis=1)
# 
#  setting up testing and training sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
# 
#  concatenate our training data back together
# X = pd.concat([X_train, y_train], axis=1)
# 
#  separate minority and majority classes
# not_fraud = X[X.Class==0]
# fraud = X[X.Class==1]
# 
# upsample minority
# fraud_upsampled = resample(fraud,
#                           replace=True, # sample with replacement
#                           n_samples=len(not_fraud), # match number in majority class
#                           random_state=27) # reproducible results
# 
# # combine majority and upsampled minority
# upsampled = pd.concat([not_fraud, fraud_upsampled])
# 
# # check new class counts
# upsampled.Class.value_counts()
#     1    213245
#     0    213245

# In[32]:





# In[33]:


# Here we are oversampling due to imbalanced dataset as undepressed is more than depressed

# we reset the index and drop the ID and DATE columns
X = unified_df.reset_index().drop(["ID", 'DATE'], axis=1)
#separate minority and majority classes
un_depressed = X[X.LABEL==0]
depressed = X[X.LABEL==1]

#upsample minority
depressed_upsampled = resample(depressed,
                          replace=True, # sample with replacement
                          n_samples=len(un_depressed), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([un_depressed, depressed_upsampled])

# check new class counts
upsampled.LABEL.value_counts()
#     1    213245
#     0    213245


# In[34]:


#upsampled




def is_valid_token(token):
    if token.is_punct:
        return False
    if token.is_stop:
        return False
    # if token.text.lower() in STOPWORDS:
    #     return False
    if not token.is_alpha:
        return False
    return True


def spacy_tokenizer(answer):
    
    doc = nlp(answer)
    
    # Creating our token object, which is used to create documents with linguistic annotations.
    tokens = [token for token in doc if is_valid_token(token)]
    print("tokens 1 = " +str(tokens) )

    # Lemmatizing each token and converting each token into lowercase
    tokens = [ word for word in tokens if not word in english_stopwords_spacy and word.text not in punctuations ]
    print("tokens 2 = " + str(tokens))
    #tokens = [token.text for token in tokens]

    # Removing stop words
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
    print("tokens 3 = " + str(tokens))
    # return preprocessed list of tokens
    return tokens

resulted_token  = spacy_tokenizer(str(upsampled['TITLE_TEXT'][0]))
print("resulted tokens = " + str(resulted_token))