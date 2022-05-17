

class StructEmbed(nn.Module):
    def __init__(self, node_features, edge_features,
        hidden_dim, out_dim = 16, num_encoder_layers=3, 
        vocab=20, k_neighbors=30, protein_features='full', augment_eps=0.,
        dropout=0.1, use_mpnn=False):
        """ Graph labeling network """
        super(StructEmbed, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Featurization layers
        self.features = ProteinFeatures(
            node_features, edge_features, top_k=k_neighbors,
            features_type=protein_features, augment_eps=augment_eps,
            dropout=dropout
        )

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        layer = TransformerLayer if not use_mpnn else MPNNLayer

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.hidden_out = nn.Linear(hidden_dim, out_dim, bias=True)

       
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, X, S, L, mask):
        """ Graph-conditioned sequence model """

        # Prepare node and edge embeddings
        V, E, E_idx = self.features(X, L, mask)
        h_V = self.W_v(V)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV, mask_V=mask, mask_attend=mask_attend)

        node_embeddings = self.hidden_out(h_V)

        return node_embeddings