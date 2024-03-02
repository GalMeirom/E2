import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import utils as ut

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length, numC):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        self.numC = numC
        if numC != 1:
            self.pred1 = nn.Linear(hidden_size, hidden_size)
            self.pred2 = nn.Linear(hidden_size, numC)

    
    def predict(self, x):
        temp = x.reshape(1,-1)
        output = nn.functional.softmax(self.pred(temp), dim= 1)
        return output

    def forward(self, x):
        # Encoder
        a, (hidden, _) = self.encoder(x)
        # Latent representation
        latent = hidden[-1]  # Take the last hidden state ---> [batch size ,seq length ,hidden_size]
        # Decoder
        #latent.unsqueeze(1).repeat(1,x.size(1),1)
        decoded, (hidden_gal, l) = self.decoder(latent.unsqueeze(1).repeat(1,x.size(1),1)) # -----> [batch size, seq length, hidden size]
        latent_pred = hidden_gal[-1]
        rec_output = torch.tanh(self.linear(decoded))# -----> [batch size, seq length, input size]

        if  self.numC != 1:
            # -----> [batch size, num of classes]
            result_list = [self.predict(row) for row in decoded]
            pred = torch.stack(result_list, dim=1).squeeze(0)
            # pred = nn.functional.softmax(torch.tanh(self.pred2(torch.tanh(self.pred1(latent_pred)))), dim= 1)

            return rec_output, pred
        else:
            return rec_output, _

class LSTM_Model():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate, clip_value, seq_length, numC):
        self.model = LSTMAutoencoder(input_size, hidden_size, num_layers, seq_length, numC)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip = clip_value
        self.input_size = input_size
    
    def train(self, train_dataloader, epochs):
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch in train_dataloader:
                batch_data = batch[0]
                if len(batch_data.shape) == 4:
                    batch_data = batch_data.squeeze(1)
                output, _ = self.model(batch_data)
                loss = self.criterion(output, batch_data)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_dataloader)
            #if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], AVG Training Loss: {avg_train_loss}')

    def eval(self, val_dataloader):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0]
                if len(batch_data.shape) == 4:
                    batch_data = batch_data.squeeze(1)
                output, (h, c) = self.model(batch_data)
                val_loss = self.criterion(output, batch_data)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f'AVG VAL Loss: {avg_val_loss}')
        return avg_val_loss
    

class LSTM_Model_wPred():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate, clip_value, seq_length, numC):
        self.model = LSTMAutoencoder(input_size, hidden_size, num_layers, seq_length, numC)
        self.criterionMSE = nn.MSELoss()
        self.criterionCE = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.clip = clip_value
        self.input_size = input_size
    
    def train(self, train_dataloader):
            self.model.train()
            reco_train_loss = 0.0
            label_train_loss = 0.0
            for batch in train_dataloader:
                batch_data = batch[0]
                batch_label = batch[1]
                if len(batch_data.shape) == 4:
                    batch_data = batch_data.squeeze(1)
                output, pred = self.model(batch_data)
                loss1 = self.criterionMSE(output, batch_data)
                labels = nn.functional.one_hot(batch_label, self.model.numC).type(torch.float)
                loss2 = self.criterionCE(pred, labels)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss2.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                reco_train_loss += loss1.item()
                label_train_loss += loss2.item()
            return label_train_loss / len(train_dataloader), reco_train_loss / len(train_dataloader)
            

    def eval(self, val_dataloader):
        self.model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                total_val_loss = 0.0
                batch_data = batch[0]
                if len(batch_data.shape) == 4:
                    batch_data = batch_data.squeeze(1)
                output, (h, c) = self.model(batch_data)
                val_loss = self.criterionMSE(output, batch_data)
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f'AVG VAL Loss: {avg_val_loss}')
            return avg_val_loss
    
    def pred(self, val_dataloader):
        self.model.eval()
        label_train_loss = 0.0
        succ = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0]
                batch_label = batch[1]
                if len(batch_data.shape) == 4:
                    batch_data = batch_data.squeeze(1)
                output, pred = self.model(batch_data)
                labels = nn.functional.one_hot(batch_label, self.model.numC).type(torch.float)
                loss2 = self.criterionCE(pred, labels)
                for i, pic in enumerate(pred):
                    if torch.argmax(pic) == batch_label[i]:
                        succ = succ +1
                label_train_loss += loss2.item()
            avg_label_train_loss = label_train_loss / len(val_dataloader)
            succ_rate = succ / len(val_dataloader)
            return succ_rate, avg_label_train_loss
            

