import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import utils as ut

class FullyConnectedNet(nn.Module):
    def __init__(self, layer_dimensions):
        super(FullyConnectedNet, self).__init__()

        layers = []
        for i in range(len(layer_dimensions) - 1):
            layers.append(nn.Linear(layer_dimensions[i], layer_dimensions[i+1]))
            # Add tanh activation function for all layers except the last one
            if i < len(layer_dimensions) - 2:
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length, numC, apprx):
        super(LSTMAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        # Decoder
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
        self.linear = FullyConnectedNet([hidden_size, input_size])
        self.numC = numC
        self.apprx = apprx
        if numC != 1:
            self.pred1 = nn.Linear(hidden_size , numC)
        
        if apprx != 0:
            self.approx = nn.Linear(hidden_size , numC)

    def set_apprx(self, b):
        self.apprx = b


    def forward(self, x):
        # Encoder
        x = x.to(torch.double)
        a, (hidden, _) = self.encoder(x)
        # Latent representation
        latent = hidden[-1]  # Take the last hidden state ---> [batch size ,seq length ,hidden_size]
        # Decoder
        decoded, _ = self.decoder(latent.unsqueeze(1).repeat(1,x.size(1),1)) # -----> [batch size, seq length, hidden size]
        rec_output = self.linear(decoded)# -----> [batch size, seq length, input size]

        if  self.numC != 1:
            pred = nn.functional.softmax(torch.tanh(self.pred1(latent)), dim= 1)
            return rec_output, pred
        if  self.apprx != 0:
            if self.apprx == 1:
                list_of_prefix = [x[:, :i, :] for i in range(1, x.size(1) + 1)]
                self.set_apprx(0)
                right_list = []
                for pref in list_of_prefix:
                    _, (hid, t) = self.encoder(pref)
                    pref_latent = hid[-1]
                    pred = torch.tanh(self.approx(pref_latent))
                    right_list.append(pred)
                app = torch.stack(right_list, dim= 1)
                self.set_apprx(1)
                return rec_output, app
            else:
                size = x.size(1)
                acc, _ = torch.split(x, size / 2, dim=1)
                curr = acc.clone()
                self.set_apprx(0)
                while acc.size(1) < size:
                    _, (hid, t) = self.encoder(curr)
                    acc_latent = hid[-1]
                    pred = torch.tanh(self.approx(acc_latent))
                    curr = torch.stack([curr, pred], dim= 1)
                    curr = torch.cat((curr[:, :0, :], curr[:, 1:, :]), dim=1)
                    acc = torch.stack([acc, pred], dim=1)
                self.set_apprx(2)
                return rec_output, acc
        else:
            return rec_output, _
            
        
        

class LSTM_Model():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate, clip_value, seq_length, numC, apprx):
        self.model = LSTMAutoencoder(input_size, hidden_size, num_layers, seq_length, numC, apprx)
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
            avg_train_loss = total_train_loss / len(train_dataloader.dataset)
            #if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], AVG Training Loss: {avg_train_loss}')
            #if epoch % 20 == 0:
                #ut.reconstruct(self, train_dataloader.dataset.tensors[0])

    def eval(self, val_dataloader):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0]
                if len(batch_data.shape) == 4:
                    batch_data = batch_data.squeeze(1)
                output, _ = self.model(batch_data)
                val_loss = self.criterion(output, batch_data)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader.dataset)
        print(f'AVG VAL Loss: {avg_val_loss}')
        return avg_val_loss
    

class LSTM_Model_wPred():
    def __init__(self, input_size, hidden_size, num_layers, learning_rate, clip_value, seq_length, numC, apprx):
        self.model = LSTMAutoencoder(input_size, hidden_size, num_layers, seq_length, numC, apprx)
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
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                reco_train_loss += loss1.item()
                label_train_loss += loss2.item()
            return label_train_loss / len(train_dataloader.dataset), reco_train_loss / len(train_dataloader.dataset)
            

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
            avg_val_loss = total_val_loss / len(val_dataloader.dataset)
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
            avg_label_train_loss = label_train_loss / len(val_dataloader.dataset)
            succ_rate = succ / len(val_dataloader.dataset)
            return succ_rate, avg_label_train_loss
    
    def train_approx(self, train_dataloader):
            self.model.train()
            train_loss = 0.0
            for batch in train_dataloader:
                batch_data = batch[0]
                next = batch_data[:, :, 1:2]
                batch_data = batch_data[:, :, 0:1]
                output, approx = self.model(batch_data)
                loss1 = self.criterionMSE(output, batch_data)
                loss2 = self.criterionMSE(approx, next)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                train_loss = train_loss + loss1.item() + loss2.item()
                
            return train_loss / len(train_dataloader.dataset)
    
    def val_approx(self, val_dataloader):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch_data = batch[0]
                next = batch_data[:, :, 1:2]
                batch_data = batch_data[:, :, 0:1]
                output, approx = self.model(batch_data)
                loss1 = self.criterionMSE(output, batch_data)
                loss2 = self.criterionMSE(approx, next)
                total_val_loss = total_val_loss + loss1.item() + loss2.item()
            avg_val_loss = total_val_loss / len(val_dataloader.dataset)
            return avg_val_loss
            

