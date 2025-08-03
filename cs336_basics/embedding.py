import torch 
import torch.nn as nn 

class Embedding(nn.Module):
    def __init__(self, 
        num_embeddings: int, # Size of the vocabulary
        embedding_dim: int, # Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None, # Device to store the parameters on
        dtype: torch.dtype | None = None # Data type of the parameters
    ): 
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device 
        self.dtype = dtype
        self.embed_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.init_embed_matrix()

    def init_embed_matrix(self):
        nn.init.trunc_normal_(self.embed_matrix, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_matrix[token_ids]