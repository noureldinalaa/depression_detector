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
                    if file.is_file():
                        frame = self.xml_parsing(file)
                        frames.append(frame)

        return frames
