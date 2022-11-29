import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        print('modules ='+str(modules))
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
       
        
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0.25, 
                             batch_first=True
                           )
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.embedding_layer.weight)
        
    def forward(self, features, captions):
        captions = captions[:, :-1] 
        embed = self.embedding_layer(captions) 
        
        inputs = torch.cat((features.unsqueeze(1), embed), dim=1)
        lstm_outputs, _ = self.lstm(inputs)
        outputs = self.linear(lstm_outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        predictions = []
        count = 0
        word_idx = None
        
        while count < max_len and word_idx != 1 :
            
            output_lstm, states = self.lstm(inputs, states)
            output = self.linear(output_lstm)
            
            _, indx = output.max(2)            
            word_idx = indx.item()
            predictions.append(word_idx)
            
            inputs = self.embedding_layer(indx)
            
            count+=1
        
        return predictions