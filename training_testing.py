import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

class training_testing(object):
    def loader_creation(self,labels,features,split_frac,val_test_frac,batch_size):


        ## split data into training, validation, and test data (features and labels, x and y)
        split_idx = int(len(features) * split_frac)
        train_x, remaining_x = features[:split_idx], features[split_idx:]
        train_y, remaining_y = labels[:split_idx], labels[split_idx:]

        test_idx = int(len(remaining_x) * val_test_frac)
        val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

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