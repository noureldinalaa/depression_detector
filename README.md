# depression_detector
## Introduction

Language plays a central role for psychologists who rely on manual coding of patient language for diagnosis. Using language technology for psycholog-ical diagnosis, particularly for depression detection could lead to low-priced screening test and affordable treatment. For Human Behaviour Understanding Using Machine Learning and Psychological Methods project we explore  two  neural  architectures:   Recurrent  Neural  Net-work  (RNN)  and Bidirectional  Encoder  Representations  from  Transformer  (BERT), for  de-pression detection on Reddit posts. In this repository we will implement the data preparation and pre-procesing as well as the implementation of the RNN and BERT models for detecting depression


## Dataset Overview

Dataset used in this project is from the Early Risk Prediction on the Internet (erisk2017)  workshop.   This  dataset  provides  user-generated  content  consisting of  Reddit  posts  organized  and  processed  chronologically. Every  user  ha sreceived  a  label  as  risk  or  non-risk  (of  depression).   The  dataset  has  two parts:  a training dataset and a test dataset, and each part has ten chunks with series of XML files. These XML files store users’ posts and the respective posts’ comments. 
**For obtaining the dataset please contact the organisers in this [link](https://early.irlab.org/)** .

## Important libraries used
1. Pytoch : for training our modles.
2. SpaCy : for tokenization.
3. Skorch: library that allow us using pytorch with Skorch.

## Pretrained Model
In this project we have tried to train our model using pretrained word vectors provided by stanford university. As we are focusing on social media we have used twitter pretrained word vectors which can be downloaded from this [link](https://nlp.stanford.edu/projects/glove/) .

## Project Instructions

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/noureldinalaa/depression_detector.git
		cd depression_detector 

2. Install packages like pytorch and torch vision and some pip packages in requirements text file :
	```
		conda install pytorch torchvision -c pytorch
            pip install -r requirements.txt
	```
3. Choose which model you would like to train from **Depression_detector.py** python file, uncomment one of these models:
  - RNN Model without pretrained word vectors .
  - RNN Model With pretrained word vectors .
  - BERT 
  and then start training with 
  
  ```
    python Depression_detector.py
  ``` 
4. To train RNN using Skorch:

    ```
    python RNN_skorch.py
 
    ```
    
## Results
Accuracy in both RNN's Models was almost the same, with accuracy around 80 % on training dataset , 72% on validation dataset and 65% on unseen test dataset. While in Bert accuracy was more promising, with accuracy of 93 % on training dataset and 72% on validation dataset.Noticing that we have tried different hyperparameters and these were the highest results obtained.

**For more detailed explanation of the steps, please find it in this [link]( https://github.com/noureldinalaa/depression_detector/blob/master/project_explanation/Detecting_Depression_on_social_media.pdf) 


## Next steps

Imporve our models performance to increase the accuracy on unseen dataset.


