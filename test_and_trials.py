#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:32:45 2020

@author: nobot
"""


import os ,os.path ,sys
import numpy as np
import collections 
import xml.etree.ElementTree as et
import csv
import pandas as pd
path_p = "/home/nobot/Human_behavior/2017/train/positive_examples_anonymous_chunks"
path_n = "/home/nobot/Human_behavior/2017/train/negative_examples_anonymous_chunks"
path_test = "/home/nobot/Human_behavior/2017/test"
# l = os.listdir(path)
# print(l)



# files = []
# for f in os.listdir(path):
#     file_path = os.path.join(path, f)
#     #print(f)
#     #for root , dirs , files in os.walk(path):
#     if os.path.isfile(file_path):
#         files.append(file_path)
       
        
# print(files)

# frames = []
# for f1 in listdir_fulllpath(path_records):      
#   if os.path.isdir(f1):
#     for j in listdir_fulllpath(f1):
#       if os.path.isdir(j):
#         frames.append(parse_folder(j))
          
# print(frames)         
#     # for f in files:
#     #     print(os.path.join(root, f))

# individuos_train_p=list(os.listdir(path+'/chunk_1'))
# print(individuos_train_p)
# individuos_train_p= [individuo.replace('1.xml','') for individuo in individuos_train_p if individuo !='desktop.ini']
# print()
# print(individuos_train_p)
# print(len(individuos_train_p))
# print('\n Respuestas positvas entrenamiento')   
# for chunk in range(1,11):
#     individ_in_chunk_p=os.listdir(path+'/chunk_'+str(chunk))
#     individ_in_chunk_p=[individuo.replace(str(chunk)+'.xml','') for individuo in individ_in_chunk_p if individuo !='desktop.ini']
#     #print('chunk:',chunk,'\t',set(individuos_train_p)==set(individ_in_chunk_p))
#     #print('\t',len(individ_in_chunk_p))


#Read multiple files 
#https://www.youtube.com/watch?v=Zy9u0nNGRDg

#how to convert xml file to panda dataframe
#https://www.youtube.com/watch?v=WWgiRkvl1Ws
     
#merge xml files
#https://stackoverflow.com/questions/15921642/merging-xml-files-using-pythons-elementtree

#from xml to pandas datafeame
#https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c

def xml_parsing(xml_file):
    #xml_file = '/home/nobot/Human_behavior/2017/train/test_folder/chunk_1/train_subject96_1.xml'
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    
    subject_id = xroot.find('ID').text
    writings = []
    
    for writing in xroot.findall('WRITING'):
        xml_data = {}
        xml_data['ID'] = subject_id
        xml_data['TITLE'] = writing.find('TITLE').text
        xml_data['DATE'] = writing.find('DATE').text
        xml_data['INFO'] = writing.find('INFO').text
        xml_data['TEXT'] = writing.find('TEXT').text
        
        writings.append(xml_data)
        
    df = pd.DataFrame(writings)

    #print(df)
    #if there is no ID or Date return none
    # try:
    #   df = df
    #   #df = df.set_index(['ID', 'DATE'])
    # except:
    #   df =  None
      
    return df
  
#training     
#Positive  
def train_positive_extractor():
    subjects = []
    frames = []
    total_frames = []

    for chunk in range(1,11):
        chunks_p = os.listdir(path_p+'/chunk_'+str(chunk))
        subjects = chunks_p
        length = len(subjects)
        for i in range(length):
            xml_file = '/home/nobot/Human_behavior/2017/train/positive_examples_anonymous_chunks/chunk_' +str(chunk) + '/'+str(subjects[i])
            frames = xml_parsing(xml_file)
            total_frames.append(frames)

            # print(frames)
            
            
    DF = pd.concat(total_frames)
    shape =  DF.shape
    labels = np.ones((shape[0],1))
    print(labels.shape)
    DF['labels'] = labels
    
    DF.to_csv('train_data_p.csv', sep=',',index=False)
    return DF
#negative 
def train_negative_extractor():
    subjects = []
    frames = []
    total_frames = []
    debug = []
    for chunk in range(1,11):
        chunks_n = os.listdir(path_n+'/chunk_'+str(chunk))
        subjects = chunks_n
        length = len(subjects)
        for i in range(length):
            xml_file = '/home/nobot/Human_behavior/2017/train/negative_examples_anonymous_chunks/chunk_' +str(chunk) + '/'+str(subjects[i])
            debug.append(subjects[i])
            #print(str(subjects[i] + 'suceeded'))
            frames = xml_parsing(xml_file)
            total_frames.append(frames)

            # print(frames)
            
            
    DF = pd.concat(total_frames)
    shape =  DF.shape
    labels = np.zeros((shape[0],1))
    print(labels.shape)
    DF['labels'] = labels
    
    DF.to_csv('train_data_n.csv', sep=',',index=False)
    return DF

def pos_neg_concat(df1 , df2):
    df_list = [df1 , df2]
    concatenation = pd.concat(df_list)
    concatenation.to_csv('all_train_data.csv', sep=',',index=False)

def test_extractor():
    subjects = []
    frames = []
    total_frames = []

    for chunk in range(1, 11):
        chunks_p = os.listdir(path_test + '/chunk_' + str(chunk))
        subjects = chunks_p
        length = len(subjects)
        for i in range(length):
            xml_file = '/home/nobot/Human_behavior/2017/test/chunk_' + str(chunk) + '/' + str(subjects[i])
            frames = xml_parsing(xml_file)
            total_frames.append(frames)


            # print(frames)

    DF = pd.concat(total_frames)
    shape = DF.shape
    data_list=[]
    data_dic = {}

    with open("/home/nobot/Human_behavior/2017/test/test_golden_truth.txt") as file:  # Use file to refer to the file object
        data = file.readlines()
        for i in data:
            key = i[1:-3]
            #print(key + " " + "succeed")
            value = int(i[-2])
            data_dic.update({key:value})
        #print("finished")

    # labels = np.ones((shape[0], 1))
    # print(labels.shape)
    labels= []
    print(DF['ID'])
    for id in DF['ID']:
        label = data_dic.get(id)
        labels.append(label)
    DF['labels'] = labels

    # def createdoc(id, **kwargs):
    #     # for id in DF['ID']:
    #     if id == data_dic['
    #         data_dic[id]
    # # # print(type(answer), answer)
    # #
    # # doc = nlp(str(answer))
    # return doc
    #
    # df['labels'] = df['ID'].apply(createdoc, axis='columns')
    # df.head(3)
    # DF['labels'] = labels

    DF.to_csv('test_data.csv', sep=',',index=False)
    return DF


#train
positive_df = train_positive_extractor()
negative_df = train_negative_extractor()
pos_neg_concat(positive_df ,negative_df)

#test
test_extractor()




# print(chunks_p)
# print()

  



        
