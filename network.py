import torch
import torch.nn as nn
import random

class MlpDQN(nn.Module):
    def __init__(self, action_size, input_size):
        super(MlpDQN, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        # self.fc1 = nn.Linear(self.input_size, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, self.action_size)
        self.fc1 = nn.Linear(self.input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_size)       
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
        # torch.nn.init.uniform_(self.fc1.weight.data, a=0.1, b=0.2)
        # torch.nn.init.uniform_(self.fc2.weight.data, a=0, b=0.1)
        # torch.nn.init.uniform_(self.fc3.weight.data, a=0, b=0.1)
        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ModelPredictor(nn.Module):
    def __init__(self, output_size, input_size):
        super(ModelPredictor, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, output_size, input_size):
#         super(Encoder, self).__init__()
#         self.output_size = output_size
#         self.input_size = input_size
#         self.fc1 = nn.Linear(self.input_size, 24)
#         self.fc2 = nn.Linear(24, 24)
#         self.fc3 = nn.Linear(24, self.action_size)
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class Encoder_rnn(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(input_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size, input dim]
        
        outputs, hidden = self.rnn(self.dropout(src))
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden

class Decoder_rnn(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.RNN(output_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        input = self.dropout(input)
                
        output, hidden = self.rnn(input, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size, tau_dim]
        #trg = [trg len, batch size, tau_dim]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_tau_size = trg.shape[2]

        
        #tensor to store decoder outputs
        outputs = torch.zeros_like(trg)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        z_t = hidden
        
        input = trg[0,:]
        outputs[0] = input
        mask = (trg != 0).float().to(self.device)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden)
            output = output * mask[t]
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else output
        
        return outputs, z_t