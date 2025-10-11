import torch
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import MLP, GCN, GAT, PNA, global_max_pool, AttentionalAggregation
from torch.nn import Linear
from torch import nn
from global_config import global_config


import warnings
warnings.filterwarnings('ignore', category=UserWarning)


class GNN(torch.nn.Module):
    """
    GNN class to predict jammer coordinates based on network data.

    This GNN model can be configured to use various types of graph convolutional layers, including
    Multi-Layer Perceptrons (MLP), Graph Convolutional Network (GCN), Graph Attention Network (GAT),
    and Principal Neighbors Aggregation (PNA). The model architecture includes configurable dropout
    and fully connected layers for regression of the target features.

    Args:
        in_channels (int): Number of input features per node.
        dropout_rate (float): Dropout rate used for regularization to prevent overfitting.
        num_heads (int): Number of attention heads in GAT layers, used only if GAT is selected as model_type.
        model_type (str): Type of GNN to use ('MLP', 'GCN', 'GAT', 'GATv2', 'PNA').
        hidden_channels (int): Number of hidden units in each GNN layer.
        out_channels (int): Number of output features from the last GNN layer before regression.
        num_layers (int): Number of layers in the GNN.
        act (str, optional): Activation function to use (default 'relu').
        norm (str, optional): Type of normalization layer to use, if any (default None).
        deg (np.ndarray, optional): Degree information for nodes, used only in PNA for scalable aggregations (default None).

    Attributes:
        gnn (torch.nn.Module): The graph neural network module, varies based on the `model_type`.
        regressor (torch.nn.Linear): Linear layer to regress the graph-level features to target outputs.
        weight_layer (torch.nn.Sequential): MLP for L_adapt confidence vector output.
        dropout (torch.nn.Dropout): Dropout layer applied to the outputs of the GNN.

    Methods:
        forward(input): Defines the forward pass of the model using input data.
    """
    def __init__(self, in_channels, dropout_rate, num_heads, model_type, hidden_channels, out_channels, num_layers, act='relu', norm=None, deg=None):
        super(GNN, self).__init__()

        # Model definitions
        if model_type == 'MLP':
            self.gnn = MLP(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type == 'GCN':
            self.gnn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm)
        elif model_type in ['GAT', 'GATv2']:
            self.gnn = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=0.0, act=act, norm=norm, heads=num_heads, v2='v2' in model_type)
        elif model_type == 'PNA':
            self.gnn = PNA(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers,aggregators=['mean', 'max', 'std'], scalers=['identity'], dropout=0.0, act=act, norm=None, deg=deg, jk=None)

        self.attention_pool = AttentionalAggregation(gate_nn=Linear(out_channels, 1))
        # Linear layer to regress the graph-level features to target outputs
        self.regressor = Linear(out_channels, global_config.args.out_features)

        # MLP for L_adapt confidence vector output
        self.weight_layer = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, global_config.args.out_features)
        )

        # Dropout
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Initialize weights
        init_weights(self)

    def pooling(self, x, batch):
        """Apply max pooling method."""
        return global_max_pool(x, batch)

    def forward(self, data):
        """
        Performs the forward pass of the GNN to predict the jammer's position by combining the GNN-based predictions
        with a Weighted Centroid Localization (WCL) prior using a learned confidence weight mechanism.

        The function processes input data to compute the GNN-based prediction of jammer coordinates and combines it
        with a WCL-based prediction. This combination is weighted by a confidence vector dynamically learned from
        supernode features, facilitating an adaptive balance between GNN-derived and domain-informed estimates.

        The process involves:
        - Applying the GNN model (GAT).
        - Extracting features from supernodes and applying dropout (configurable).
        - Computing a pooled graph representation for the GNN output excluding supernode.
        - Calculating the final GNN prediction and the weighted sum with the WCL prediction.

        Args:
            data (Data): PyTorch-Geometric data object containing node features (x), edge indices (edge_index),
                         edge weights (edge_weight), and WCL predictions (wcl_pred).

        Returns:
            tuple: Contains three elements:
                - gnn_prediction (Tensor): The GNN's prediction of the jammer's position.
                - final_prediction (Tensor): The element-wise weighted sum of the GNN prediction and WCL prior.
                - weight (Tensor): The learned confidence weights applied to the GNN prediction with sigmoid activation.
        """
        x, edge_index, edge_weight, wcl_pred = data.x, data.edge_index, data.edge_weight, data.wcl_pred

        # Handle based on the type of GNN
        if isinstance(self.gnn, GCN):
            x = self.gnn(x, edge_index, edge_weight=edge_weight)
        elif isinstance(self.gnn, GAT):
            x = self.gnn(x, edge_index, edge_attr=edge_weight)
        else:
            x = self.gnn(x, edge_index)

        # Extract supernode features
        if data.batch is not None:  # Batch of graphs
            # Get indices of supernodes
            supernode_indices = torch.cat([torch.where(data.batch == i)[0][-1:] for i in data.batch.unique()])
            super_node_features = x[supernode_indices]

            # Remove supernodes from x and batch
            mask = torch.ones(x.size(0), dtype=torch.bool)
            mask[supernode_indices] = False
            x = x[mask]
            batch = data.batch[mask]

        # Apply pooling to get graph representation
        if batch is not None:  # Batch of graphs
            graph_representation = self.pooling(x, batch)

        # Compute GNN-based prediction
        gnn_prediction = self.regressor(graph_representation)

        # Compute weight using supernode features
        weight = torch.sigmoid(self.weight_layer(super_node_features))

        # Compute final prediction as a weighted sum of GNN and WCL results
        final_prediction = weight * gnn_prediction + (1 - weight) * wcl_pred

        return gnn_prediction, final_prediction, weight
