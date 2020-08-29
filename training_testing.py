import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

class training_testing(object):
    def loader_creation(self,training_features,training_labels,testing_features,testing_labels,split_frac,batch_size):

        '''
        split the data anch convert it from numpy to torch

        :param training_features:
        :param training_labels:
        :param testing_features:
        :param testing_labels:
        :param split_frac:
        :param batch_size:
        :return:
        '''


        ## split data into training, validation, and test data (features and labels, x and y)
        split_idx = int(len(training_features) * split_frac)
        train_x, val_x = training_features[:split_idx], training_features[split_idx:]
        train_y, val_y = training_labels[:split_idx], training_labels[split_idx:]

        #test_data
        test_x = testing_features[:]
        test_y = testing_labels[:]

        # create Tensor datasets
        train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

        # make sure to SHUFFLE your data
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,
                                  drop_last=True)  # we put drop last in case the data we have can't be divided on batch size
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,
                                  drop_last=True)  # we put drop last in case the data we have can't be divided on batch size
        test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,
                                 drop_last=True)  # we put drop last in case the data we have can't be divided on batch size


        return train_loader,valid_loader,test_loader


    def gpu_check(self):

        '''
        check if cuda is available or not

        :return:
        '''

        # First checking if GPU is available
        train_on_gpu = torch.cuda.is_available()

        if (train_on_gpu):
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        return train_on_gpu