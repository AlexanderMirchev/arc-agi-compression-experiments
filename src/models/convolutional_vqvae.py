import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.augmentations import augment_grid_pairs
from data.scale_processing import scaling, reverse_scaling
from models.abstract_vqvae import AbstractVQVAE
from models.pipeline import Pipeline
from utils.load_data import get_grids
from utils.train_vqvae import train, validate
from utils.view import plot_losses

class ConvolutionalVQVAE(AbstractVQVAE):    
    def __init__(self, in_channels=10, starting_filters=64,
                 num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super(ConvolutionalVQVAE, self).__init__()
        
        self.starting_filters = starting_filters
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, starting_filters, kernel_size=3, stride=2, padding=1),  # 30x30 -> 15x15
            nn.BatchNorm2d(starting_filters),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(starting_filters, starting_filters*2, kernel_size=3, stride=2, padding=1),  # 15x15 -> 8x8
            nn.BatchNorm2d(starting_filters*2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(starting_filters*2, embedding_dim, kernel_size=3, stride=1, padding=1), # 8x8 -> 8x8
            nn.BatchNorm2d(embedding_dim),
            nn.LeakyReLU(inplace=True),
        )
        
        self.codebook = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, decay=0.99)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, starting_filters*2, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(starting_filters*2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(starting_filters*2, starting_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(starting_filters),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(starting_filters, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        z_e = self.encoder(x)
        return z_e
    
    def quantize(self, z_e):
        z_q, vq_loss, perplexity = self.codebook(z_e)
        return z_q, vq_loss, perplexity
    
    def decode(self, z_q):
        x_recon = self.decoder(z_q)
        return torch.clamp(x_recon, 0.0, 1.0)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.t())
            + torch.sum(self.embedding**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)

        quantized = torch.matmul(encodings, self.embedding).view(input_shape)

        if self.training:
            encodings_sum = encodings.sum(0)
            ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * encodings_sum
            dw = torch.matmul(encodings.t(), flat_input)
            ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            self.ema_cluster_size = ema_cluster_size
            self.ema_w = ema_w

            n = ema_cluster_size.sum()
            cluster_size = ((ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon)) * n

            self.embedding.data = self.ema_w / cluster_size.unsqueeze(1)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity
    
def codebook_usage(model, data_loader, device):
    model.eval()
    codebook = model.codebook
    usage_count = torch.zeros(codebook.num_embeddings, device=device)
    total_encodings = 0

    with torch.no_grad():
        for input, output in data_loader:
            in_out = torch.cat((input, output), dim=0).to(device)

            z_e = model.encode(in_out) 

            # Flatten to (B*H*W, C)
            z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, codebook.embedding_dim)

            # Compute distances to embeddings
            distances = (
                z_e_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * z_e_flat @ codebook.embedding.t()
                + codebook.embedding.pow(2).sum(dim=1)
            )

            encoding_indices = torch.argmin(distances, dim=1)

            usage_count += torch.bincount(encoding_indices, minlength=codebook.num_embeddings).float()
            total_encodings += encoding_indices.numel()

    usage_percent = usage_count / total_encodings * 100
    active_codes = (usage_count > 0).sum().item()

    print(f"Active codebook vectors: {active_codes}/{codebook.num_embeddings} ({active_codes / codebook.num_embeddings * 100:.2f}%)")
    
    return usage_percent
    
def preprocess_grid(grid):
    grid = scaling(grid, height=30, width=30, direction='norm')

    num_classes = 10
    grid_tensor = torch.tensor(grid, dtype=torch.long)
    grid_tensor = torch.clamp(grid_tensor, 0, num_classes - 1)
    one_hot = F.one_hot(grid_tensor, num_classes=10).permute(2, 0, 1).float()
    return one_hot

def postprocess_grid(grid, grid_original):
    _, grid = torch.max(grid, dim=0)
    grid = grid.detach().cpu().numpy()
    return reverse_scaling(grid_original, grid)

def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    training_data, validation_data = get_grids(filepath="data/training")

    training_grid_pairs = [pair for task in training_data.values() for pairs in task.values() for pair in pairs]
    validation_grid_pairs = [pair for task in validation_data.values() for pairs in task.values() for pair in pairs]

    model = ConvolutionalVQVAE(
        in_channels=10, 
        starting_filters=64, 
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25
    ).to(device)
    
    print(f"Model architecture: {model}")

    training_grid_pairs = augment_grid_pairs(training_grid_pairs, target_count=15000)
    print(f"Loaded {len(training_grid_pairs)} (after augmentation) training grid pairs and {len(validation_grid_pairs)} validation grid pairs.")

    pipeline = Pipeline(
        model=model,
        preprocess_fn=preprocess_grid,
        postprocess_fn=postprocess_grid,
    )

    batch_size = 16
    train_loader = pipeline.create_data_loader(training_grid_pairs, batch_size=batch_size, shuffle=True)
    val_loader = pipeline.create_data_loader(validation_grid_pairs, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    
    max_epochs = 100
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    train_losses = []
    val_losses = []
    for epoch in range(1, max_epochs + 1):
        try:
            train_loss = train(model, 
                                train_loader, 
                                optimizer=optimizer, 
                                device=device, 
                                epoch=epoch)
            
            val_loss = validate(model, 
                                val_loader, 
                                device=device, 
                                epoch=epoch)
            
            codebook_usage(model, train_loader, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'checkpoints/conv_vqvae_batchnorm_epoch.pt')

            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss. Best: {best_val_loss}")
                break
        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            continue
    
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()