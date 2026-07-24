import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Save tensor as model state

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)



class DeepONet_BENO(nn.Module):
    #Appendix A.2 for hyperparameters
    def __init__(self, branch_input_dim, latent_dim, trunk_input_dim=2, hidden_dim=64, n_heads=2, channels=1):
        """
        branch_input_dim: Number of observations in a subdomain e.g. 32x32 flattened
        trunk_input_dim: 2 for x and y
        latent_dim: dim of the embedings before the dot product ??
        output_dim: Num of physical variables to predict (here one the velocity u)
        """
        super().__init__()
        self.channels = channels

        #Boundary transformer BENO
        self.beno_conv = nn.Conv1d(in_channels=2 + channels, out_channels=hidden_dim, kernel_size=3, padding='same')
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Project transformer output to latent dim
        self.beno_proj = nn.Linear(hidden_dim, latent_dim * channels)

        # Branch network (Now it takes interior + boundary embeddings)
        # We concatenate branch output and embeddings
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim * channels)
        )
        # Trunk network
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Glorot Normalization Appendix A.2
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u_branch, y_trunk, boundary_u):
        """
        u_branch: (batch_size, branch_input_dim)
        y_trunk: (num_points, trunk_input_dim)
        boundary_u: (batch_size, block_size*4 -4, 1)
        """

        # Conv1D expect (batch_size, channels, length)
        beno_conv_out = self.beno_conv(boundary_u.transpose(1,2)).transpose(1,2) # (batch, boudary_size, hidden_dim)

        beno_embed = self.transformer_encoder(self.pos_encoder(beno_conv_out)) # (batch, boudary_size, hidden_dim)

        #average over the seq_len to get a fixed size vector
        beno_embed = beno_embed.mean(dim=1) # (batch, hidden_dim)
        beno_embed = self.beno_proj(beno_embed)

        # branch and trunk
        branch_out = self.branch(u_branch).view(u_branch.shape[0], self.channels, -1)
        trunk_out = self.trunk(y_trunk)

        #combined branch
        beno_embed = beno_embed.view(u_branch.shape[0], self.channels, -1)
        combined_branch = branch_out + beno_embed

        dot_product = torch.einsum('bcl,pl->bcp', combined_branch, trunk_out)

        return dot_product.reshape(u_branch.shape[0], -1)

class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # num of fourier modes to keep in dim 1
        self.modes2 = modes2 # num of fourier mdoes to keep in dim 2

        self.scale = 1 / (in_channels*out_channels)
        #model parameters
        self.weight1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def mul2d(self, input, output):
        # (batch, in_channel, x, y) * (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum('bixy, ioxy -> boxy', input, output)

    def forward(self, x):
        batch_size = x.shape[0]

        # fourier transform
        x_ft = torch.fft.rfft2(x) # real fft
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weight1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weight2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO_Block(nn.Module):
    def __init__(self, width, modes1, modes2, activation=True):
        super().__init__()
        self.activation = activation
        self.conv = SpectralConv(width, width, modes1, modes2)
        #Bypass linear transformation (1x1 conv)
        self.w = nn.Conv2d(width, width, 1)
        if self.activation:
            self.act = nn.GELU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        return self.act(x1 +x2) if self.activation else x1 + x2

class FNO_BENO(nn.Module):
    def __init__(self, modes=8, width=64, num_layers=4, field_channels=1, out_channels=1):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.num_layers = num_layers
        self.channels = field_channels

        #Boundary transformer BENO
        self.beno_conv = nn.Conv1d(in_channels=2 + field_channels, out_channels=width, kernel_size=3, padding='same')
        self.pos_encoder = PositionalEncoding(d_model=width)
        encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.beno_proj = nn.Linear(width, width)


        #FNO arch
        # input is 1 channel u_velocity + 2 channels x and y
        self.fc0 = nn.Linear(2 + field_channels, self.width)
        # Stack FNO layers
        self.layers = nn.ModuleList([FNO_Block(self.width, self.modes1, self.modes2) for _ in range(num_layers -1)])
        self.last_block = FNO_Block(self.width, self.modes1, self.modes2, activation=False)
        # Project down
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels) # out 1 channel u_velocity

    def forward(self, x, grid, boundary_u):
        """
        x: (batch_size, 1, x, y)
        grid: (batch, 2, x, y)
        boundary_u: (batch_size, block_size*4 -4, 1)
        """

        # Process boundary
        beno_conv_out = self.beno_conv(boundary_u.transpose(1,2)).transpose(1,2) # (batch, boudary_size, width)
        beno_embed = self.transformer_encoder(self.pos_encoder(beno_conv_out)) # (batch, boudary_size, width)
        beno_embed = beno_embed.mean(dim=1) # (batch, width)
        beno_embed = self.beno_proj(beno_embed) # (batch, width)

        #Concat input and grid
        x_cat = torch.cat((x, grid), dim=1) # (batch, 3, x, y)
        x_cat = x_cat.permute(0, 2, 3, 1) # (batch, x, y, 3)
        v0 = self.fc0(x_cat) # (batch, x, y, width)

        # Add boundary embeddings
        v0 = v0 + beno_embed.unsqueeze(1).unsqueeze(1)
        # Permute back for spectral conv
        v0 = v0.permute(0, 3, 1, 2)

        # Apply FNO layers
        v = v0
        for layer in self.layers:
            v = layer(v)
        v = self.last_block(v)

        # Project to physical space
        v = v.permute(0, 2, 3, 1)
        v = F.gelu(self.fc1(v))
        out = self.fc2(v) # (batch, x, y, 1)

        return out.permute(0, 3, 1, 2)
