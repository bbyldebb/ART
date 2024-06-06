import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -------------------- Neural Network Architecture -------------------------
# Transformer - Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, in_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim*2,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, features):
        # (batch_size, sequence_length, hidden_size)
        # print(x.shape)
        h = F.leaky_relu(self.transformer_encoder(features))
        return h

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, dropout, norm):
        super(GraphSAGEEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        hidden_dim = hidden_dim if num_layers > 1 else out_dim
        self.input_conv = dgl.nn.GraphConv(in_dim, hidden_dim, norm=norm)
        self.convs = []
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden_dim, hidden_dim, norm=norm))
        if num_layers > 1:
            self.convs.append(dgl.nn.GraphConv(hidden_dim, out_dim, norm=norm))

    def forward(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        h = self.dropout(h)
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
            h = self.dropout(h)
        return h
    
    def transform(self, g, features):
        h = F.leaky_relu(self.input_conv(g, features))
        for conv in self.convs:
            h = F.leaky_relu(conv(g, h))
        return h

class Extractor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(Extractor, self).__init__()
        self.TFEncoder = TransformerEncoder(tf_in_dim, num_heads, tf_layers)
        self.GRUEncoder = nn.GRU(gnn_in_dim, gru_hidden_dim, gru_layers, bias=False, batch_first=True)
        self.GraphEncoder = GraphSAGEEncoder(gru_hidden_dim, gnn_hidden_dim, gnn_out_dim, gnn_layers, dropout, norm='none')
        
    def forward(self, g, features):
        bacth_size, series_len, instance_num, channel_dim = features.shape # 2,5,46,130
        h = features.permute(0,1,3,2)
        h = h.view(-1, channel_dim, instance_num)
        h = self.TFEncoder(h)
        h = h.permute(0,2,1).view(bacth_size, series_len, instance_num, channel_dim).permute(0,2,1,3).reshape(-1, series_len, channel_dim) # 92,5,130
        output, h_n = self.GRUEncoder(h)
        h = F.leaky_relu(h_n[-1]) # 92,32
        h = self.GraphEncoder(g, h) # 92, 32
        h = h.view(bacth_size, instance_num, -1) # 2,46,32
        return h
    
class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        
    def forward(self, features):
        h = F.leaky_relu(self.mlp(features))
        return h

class AutoRegressor(nn.Module):
    def __init__(self, tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout=0, tf_layers=1, gnn_layers=2, gru_layers=1):
        super(AutoRegressor, self).__init__()
        self.extractor = Extractor(tf_in_dim, num_heads, gnn_in_dim, gnn_hidden_dim, gnn_out_dim, gru_hidden_dim, dropout, tf_layers, gnn_layers, gru_layers)
        self.regressor = Regressor(gru_hidden_dim, gnn_in_dim)
        
    def forward(self, g, features):
        z = self.extractor(g, features)
        h = self.regressor(z)
        return z, h

# end Neural Network Architecture -----------------------------------------



# -------------------- Collate Function  -------------------------
# Creating a DataLoader for auto_regressor.
def collate_AR(samples):
    timestamps, graphs, feats, targets = map(list, zip(*samples))
    batched_ts = torch.stack(timestamps)
    batched_graphs = dgl.batch(graphs)
    batched_feats = torch.stack(feats)
    batched_targets = torch.stack(targets)
    return  batched_ts, batched_graphs, batched_feats, batched_targets

def create_dataloader_AR(samples, window_size=6, max_gap=60, batch_size=2, shuffle=False):
    # sliding_time_windows
    series_samples = [samples[i:i+window_size] for i in range(len(samples) - window_size + 1)]
    series_samples = [
        series_sample for series_sample in series_samples
            if all(abs(series_sample[i][0] - series_sample[i+1][0]) <= max_gap 
                for i in range(len(series_sample) - 1))
    ]
    # create a dataloader
    dataset = [[
            torch.tensor(series_sample[-1][0]),
            series_sample[-1][1], 
            torch.stack([step[2] for step in series_sample[:-1]]),
            torch.tensor(series_sample[-1][2])
        ] for _, series_sample in enumerate(series_samples)]
    dataloader = DataLoader(dataset, batch_size, shuffle, collate_fn=collate_AR)
    return dataloader

# end Collate Function  ------------------------------------------
