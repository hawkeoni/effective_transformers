import torch
import torch.nn as nn
import einops as ein


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nheads: int, dropout: float = 0.2):
        super().__init__()
        assert d_model % nheads == 0, "Number of heads should divide d_model"
        self.d_model = d_model
        self.nheads = nheads
        self.d_k = d_model // nheads
        self.Q_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.V_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Q - float tensors of shape [batch, seq_len1, d_model].
        K, V - float tensors of shape [batch, seq_len2, d_model].
        mask should be 0, where padding is.
        """
        batch_size = query.size(0)
        q_proj = self.Q_linear(query)
        q_proj = ein.rearrange(q_proj, "batch seq (heads d) -> batch heads seq d", d_k = self.d_k)
        k_proj = self.K_linear(key)
        k_proj = ein.rearrange(k_proj, "batch seq (heads d) -> batch heads d seq", d_k = self.d_k)
        v_proj = self.V_linear(value)
        v_proj = ein.rearrange(v_proj, "batch seq (heads d) -> batch heads seq d", d_k = self.d_k)
        weights = torch.matmul(q_proj, k_proj)  # batch, nheads, seq_len1, seq_len2
        weights = weights / (self.d_k ** 0.5)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e12)
        weights = torch.softmax(weights, dim=3)
        #  weights - batch, nheads, seq_len1, seq_len2
        #  V_proj - batch, nhead, seq_len2, d_k
        output = torch.matmul(weights, v_proj)  # batch, nheads, seq_len1, d_k
        output = ein.rearrange(output, "batch heads seq d -> batch seq (heads d)")
        output = output.view(batch_size, -1, self.d_model)
        output = self.out(output)
        output = self.dropout(output)
        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x - float tensor of shape [batch, seq_len, d_model].
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, ff_dim)

    def forward(self, x: torch.Tensor):
        """
        x - [batch_size, seq_len, d_model]
        """
        x = self.self_attention(x, x, x) + x
        x = self.norm1(x)
        x = self.feedforward(x) + x
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, d_model: int, num_layers: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        x - [batch_size, seq_len, d_model]
        """
        return self.layers(x)

