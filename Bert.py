from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from textwrap import wrap


class DataSetDepression(Dataset):

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
class Data_loader():
    def data_loader(self,df, tokenizer, MAX_LEN,  BATCH_SIZE):
        dataset = DataSetDepression(
            texts = df.TITLE_TEXT.to_numpy(),
            labels = df.LABEL.to_numpy(),
            tokenizer = tokenizer,
            max_len = MAX_LEN,
        )
        return DataLoader(dataset, batch_size= BATCH_SIZE, num_workers=4)

class BERTClassifier(nn.Module):

    def __init__(self, n_classes,PRE_TRAINED_MODEL_NAME = "bert-base-cased"):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, cls_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        drop_out = self.drop(cls_output)
        output = self.linear(drop_out)
        return output
class training_Bert():
    def train_model_preparation(self,model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
        model = model.train()
        losses = []
        correct_predictions = 0
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)

    def train_model(self,model,train_data_loader,loss_fn,optimizer,device,scheduler,df_train,EPOCHS):
        for epoch in range(EPOCHS):
            print('Epoch {} from {}'.format(epoch + 1, EPOCHS))
            print('----------------------------')
            train_acc, train_loss = self.train_model_preparation(
                model,
                train_data_loader,
                loss_fn,
                optimizer,
                device,
                scheduler,
                len(df_train)

            )
            print('Training: Loss {}, accuracy: {}'.format(train_loss, train_acc))
            print('')
            torch.save(model.state_dict(), 'model_trained_Bert_pretrained.pt')


class testing_Bert():
    def eval_model_preparation(self,model, data_loader, loss_fn, device, n_examples):
        model = model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
        return correct_predictions.double() / n_examples, np.mean(losses)


    def eval_model(self,model,test_data_loader,loss_fn,device,df_test,EPOCHS):
        for epoch in range(EPOCHS):
            print('Epoch {} from {}'.format(epoch + 1, EPOCHS))
            print('----------------------------')
            test_acc, test_loss = self.eval_model_preparation(
                model,
                test_data_loader,
                loss_fn, device,
                len(df_test)
            )
            print('Validation: Loss {}, accuracy: {}'.format(test_loss, test_acc))
            print('')