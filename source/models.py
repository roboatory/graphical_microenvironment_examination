import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_max_pool

class TumorGCNClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        """initializes the tumor GCN classifier

        args:
            in_features (int): number of input features
            hidden_dim (int): dimension of hidden layers
            num_classes (int): number of output classes
        """

        super(TumorGCNClassifier, self).__init__()

        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.pool = global_mean_pool
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def extract_embedding(self, x, edge_index, batch):
        """extract graph embeddings after pooling and before classification

        args:
            x (tensor): input node features
            edge_index (tensor): edge indices
            batch (tensor): batch vector
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        x = self.pool(self.conv3(x, edge_index), batch)

        embedding = torch.relu(self.fc1(x))
        
        return embedding
    
    def forward(self, x, edge_index, batch):
        """forward pass of the model

        args:
            x (tensor): input node features
            edge_index (tensor): edge indices
            batch (tensor): batch vector
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        x = self.pool(self.conv3(x, edge_index), batch)

        x = self.fc2(torch.relu(self.fc1(x)))

        return x
    
class TumorGINClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes):
        """initializes the tumor GIN classifier

        args:
            in_features (int): number of input features
            hidden_dim (int): dimension of hidden layers
            num_classes (int): number of output classes
        """

        super(TumorGINClassifier, self).__init__()
        
        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

        self.pool = global_max_pool

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def extract_embedding(self, x, edge_index, batch):
        """extract graph embeddings after pooling and before classification

        args:
            x (tensor): input node features
            edge_index (tensor): edge indices
            batch (tensor): batch vector
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        x = self.pool(self.conv3(x, edge_index), batch)

        embedding = torch.relu(self.fc1(x))
        
        return embedding
    
    def forward(self, x, edge_index, batch):
        """forward pass of the model

        args:
            x (tensor): input node features
            edge_index (tensor): edge indices
            batch (tensor): batch vector
        """
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        x = self.pool(self.conv3(x, edge_index), batch)

        x = self.fc2(torch.relu(self.fc1(x)))

        return x

class TumorGATClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim, num_classes, heads = 1):
        """initializes the tumor GAT classifier

        args:
            in_features (int): number of input features
            hidden_dim (int): dimension of hidden layers
            num_classes (int): number of output classes
            heads (int, optional): number of attention heads; \
                                   defaults to 1
        """

        super(TumorGATClassifier, self).__init__()

        self.conv1 = GATConv(in_features, hidden_dim, heads = heads, concat = True)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads = heads, concat = True)
        self.conv3 = GATConv(hidden_dim * heads, hidden_dim, heads = heads, concat = True)

        self.pool = global_mean_pool

        self.fc1 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def extract_embedding(self, x, edge_index, batch):
        """extract graph embeddings after pooling and before classification

        args:
            x (tensor): input node features
            edge_index (tensor): edge indices
            batch (tensor): batch vector
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        x = self.pool(self.conv3(x, edge_index), batch)

        embedding = torch.relu(self.fc1(x))
        
        return embedding
    
    def forward(self, x, edge_index, batch):
        """forward pass of the model

        args:
            x (tensor): input node features
            edge_index (tensor): edge indices
            batch (tensor): batch vector
        """

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        x = self.pool(self.conv3(x, edge_index), batch)

        x = self.fc2(torch.relu(self.fc1(x)))
        
        return x