# imports
import xml.etree.ElementTree as et
import pandas as pd
from pathlib import Path

class Depression_detection(object):
    # Constructor may be built while creating the project later
    def __init__(self,base_path,training_positive_path,training_negative_path,test_path):
        self.base_path = base_path
        self.training_positive_path = training_positive_path
        self.training_negative_path = training_negative_path
        self.test_path = test_path

    def xml_parsing(self,xml_file):
        ''' Parses a xml file  with the following structure

          <INDIVIDUAL>
            <ID> ... </ID>
            <WRITING>
              <TITLE> ...   </TITLE> <DATE> ... </DATE>  <INFO> ... </INFO>  <TEXT> ...  </TEXT>
            </WRITING>
            ....
          </INDIVIDUAL>

          Returns DateFrame with five columns ID TITLE DATE INFO TEXT

        '''

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

        return df

    def parse_folder(self,folder_to_parse):
        '''
        1. It parses the training /testing directory where each folder contains 10 chuncks as well as
        2. It calls the XML parser and returns list of data frames

        :param folder_to_parse:
        :return: list of frames
        '''
        frames = []
        folders_path = Path(folder_to_parse)
        for directory in folders_path.iterdir():
            if directory.is_dir():
                print(directory.name)
                for file in directory.iterdir():
                    if file.is_file() and not file.name.startswith('._'):
                        frame = self.xml_parsing(file)
                        frames.append(frame)

        return frames

    def prepare_test_dataframe(self,test_dataframe):
        '''
        Generate a dataframe that combine the test dataframe
        with the ground truth labels.

        :param test_dataframe:
        :return: test dataset with labels
        '''

        test_dataframe= test_dataframe.set_index('ID')
        test_gold_path = self.test_path.joinpath('./test_golden_truth.txt')
        golden_truth_dataframe = pd.read_csv(str(test_gold_path),names=['ID','LABEL'], sep="\t")
        print(golden_truth_dataframe.columns)
        golden_truth_dataframe['ID'] = golden_truth_dataframe.ID.apply(self.remove_space)
        golden_truth_dataframe = golden_truth_dataframe.set_index('ID')
        #This one merge both of the dataframes considereing ID's arrangment.
        golden_test_dataframe = pd.merge(test_dataframe, golden_truth_dataframe, how='left', left_index=True, right_index=True)

        return golden_test_dataframe

    def remove_space(self,subject_ID):
        '''
        It strips extra characters in the subject (like extra spaces and /n)
        :param string:
        :return: stripped subject ID
        '''
        clean_subject_ID = subject_ID.strip()
        return clean_subject_ID

    def Unifing_training_data(self,positive_training_file_CSV,negative_training_file_CSV):
        positive_training_file_df= pd.read_csv(positive_training_file_CSV, index_col=0)
        print(positive_training_file_df.shape)
        negative_training_file_df = pd.read_csv(negative_training_file_CSV, index_col=0)
        print(negative_training_file_df.shape)
        unified_df = pd.concat([positive_training_file_df, negative_training_file_df])
        return unified_df




