import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import numpy as np

import pandas as pd

class training_testing():
    def loader_creation(self,training_features,training_labels,testing_features,testing_labels,split_frac,batch_size,idx = 1):

        '''
        split the data and convert it from numpy to torch

        :param training_features:
        :param training_labels:
        :param testing_features:
        :param testing_labels:
        :param split_frac:
        :param batch_size:
        :return:
        '''



        # ##cross validation
        split_idx_1 = int(len(training_features) * 0.8)
        train_x_80, train_x_1 = training_features[:split_idx_1], training_features[split_idx_1:]
        train_y_80, train_y_1 = training_labels[:split_idx_1], training_labels[split_idx_1:]
        split_idx_2 = int(len(train_x_80) * 0.5)
        train_x_40_1, train_x_40_2 = train_x_80[:split_idx_2], train_x_80[split_idx_2:]
        train_y_40_1, train_y_40_2= train_y_80[:split_idx_2], train_y_80[split_idx_2:]
        split_idx_3 = int(len(train_y_40_1) * 0.5)
        train_x_2, train_x_3 = train_x_40_1[:split_idx_3], train_x_40_1[split_idx_3:]
        train_y_2, train_y_3= train_y_40_1[:split_idx_3], train_y_40_1[split_idx_3:]
        split_idx_4 = int(len(train_y_40_2) * 0.5)
        train_x_4, train_x_5 = train_x_40_2[:split_idx_4], train_x_40_2[split_idx_4:]
        train_y_4, train_y_5= train_y_40_2[:split_idx_4], train_y_40_2[split_idx_4:]

        def fold(idx):
            if idx ==1:
                fold_x_train = np.concatenate((train_x_2, train_x_3, train_x_4, train_x_5),axis=0)
                fold_x_valid = train_x_1
                fold_y_train = np.concatenate((train_y_2, train_y_3, train_y_4, train_y_5),axis=0)
                fold_y_valid = train_y_1
                return fold_x_train,fold_x_valid,fold_y_train,fold_y_valid

            elif idx == 2:
                fold_x_train = np.concatenate((train_x_1, train_x_3, train_x_4, train_x_5),axis=0)
                fold_x_train = pd.concat(fold_x_train )
                fold_x_valid = train_x_2
                fold_y_train = np.concatenate((train_y_1, train_y_3, train_y_4, train_y_5),axis=0)
                fold_y_valid = train_y_2
                return fold_x_train,fold_x_valid,fold_y_train,fold_y_valid

            elif idx == 3:
                fold_x_train = np.concatenate((train_x_1, train_x_2, train_x_4, train_x_5),axis=0)
                fold_x_valid = train_x_3
                fold_y_train = np.concatenate((train_y_1, train_y_2, train_y_4, train_y_5),axis=0)
                fold_y_valid = train_y_3
                return fold_x_train,fold_x_valid,fold_y_train,fold_y_valid

            elif idx == 4:
                fold_x_train = np.concatenate((train_x_1, train_x_2, train_x_3, train_x_5),axis=0)
                fold_x_valid = train_x_4
                fold_y_train = np.concatenate((train_y_1, train_y_2, train_y_3, train_y_5),axis=0)
                fold_y_valid = train_y_4
                return fold_x_train,fold_x_valid,fold_y_train,fold_y_valid

            elif idx == 5:
                fold_x_train = np.concatenate((train_x_1, train_x_2, train_x_3, train_x_4),axis=0)
                fold_x_valid = train_x_5
                fold_y_train = np.concatenate((train_y_1, train_y_2, train_y_3, train_y_4),axis=0)
                fold_y_valid = train_y_5
                return fold_x_train,fold_x_valid,fold_y_train,fold_y_valid


        ## split data into training, validation, and test data (features and labels, x and y)
        # split_idx = int(len(training_features) * split_frac)

        train_x, val_x,train_y, val_y = fold(idx)
        # train_x, val_x = training_features[:split_idx], training_features[split_idx:]
        # train_y, val_y = training_labels[:split_idx], training_labels[split_idx:]

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


    def RNN_training(self,RNN_net,lr=0.001,epochs = 5,train_on_gpu =False
                     ,batch_size=50,train_loader=3000,valid_loader=3000,criterion =0 ,optimizer=0):
        # loss and optimization functions


        print_every = 100

        clip = 5  # gradient clipping
        counter = 0
        num_correct = 0
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf

        if (train_on_gpu):
            RNN_net.cuda()

        RNN_net.train()

        # train for some number of epochs
        for e in range(epochs):
            # initialize hidden state(return all hidden states zeros)
            h = RNN_net.init_hidden(batch_size)

            # batch loop
            for inputs, labels in train_loader:
                counter += 1

                if (train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                # this can be h and c(so every time new h and c)
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                RNN_net.zero_grad()

                # get the output from the model
                output, h = RNN_net(inputs, h)

                # calculate the loss and perform backprop
                """   
                **HERE ** 
                we are making sure that our outputs are squeezed so that they 
                do not have an empty dimension output.squeeze() and 
                the labels are float tensors, labels.float(). 
                Then we perform backpropagation as usual.        
                """
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(RNN_net.parameters(), clip)
                optimizer.step()



                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = RNN_net.init_hidden(batch_size)
                    val_losses = []
                    RNN_net.eval()
                    for inputs, labels in valid_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if (train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, val_h = RNN_net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                        # convert output probabilities to predicted class (0 or 1)
                        pred = torch.round(output.squeeze())  # rounds to the nearest integer

                        # compare predictions to true label
                        correct_tensor = pred.eq(labels.float().view_as(pred))
                        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                            correct_tensor.cpu().numpy())
                        num_correct += np.sum(correct)

                    # -- stats! -- ##
                    # accuracy over all valid data
                    valid_acc = num_correct / len(valid_loader.dataset)



                    RNN_net.train()
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))
                    # torch.save(RNN_net.state_dict(), 'model_trained_RNN_not_pretrained.pt')
                    print("Validation accuracy: {:.3f}".format(valid_acc))


                    if np.mean(val_losses) <= valid_loss_min:
                        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                            valid_loss_min,
                            np.mean(val_losses)))
                        torch.save(RNN_net.state_dict(), 'model_trained_RNN_not_pretrained.pt')
                        valid_loss_min = np.mean(val_losses)

                    num_correct = 0
    def RNN_test(self,RNN_net,lr=0.001,epochs = 5,train_on_gpu =False
                 ,batch_size=50,test_loader=3000,criterion =0 ,optimizer=0):



        # Get test data loss and accuracy

        test_losses = []  # track loss
        num_correct = 0

        # init hidden state
        h = RNN_net.init_hidden(batch_size)

        RNN_net.eval()
        # iterate over test data
        for inputs, labels in test_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # get predicted outputs
            output, h = RNN_net(inputs, h)

            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer

            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            num_correct += np.sum(correct)

        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct / len(test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))

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


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5,  train_on_gpu =False):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()


        self.train_on_gpu =train_on_gpu
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define all layers
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        # dropout layer
        #self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0) #50 as example

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        # out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())

        return hidden



