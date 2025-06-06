import torch
import torch.nn.functional as F
from models.abstract_vqvae import AbstractVQVAE

def train(model: AbstractVQVAE, train_loader, optimizer, device, epoch=0):
    model.train()
    train_loss = 0
    recon_loss = 0
    vq_loss = 0
    num_batches = 0
    
    for batch_idx, (input, output) in enumerate(train_loader):
        in_out = torch.cat((input, output), dim=0)
        in_out = in_out.to(device)

        optimizer.zero_grad()
        
        try:
            recon_batch, vq_commitment_loss, _, _ = model(in_out)
            
            reconstruction_loss = F.binary_cross_entropy(recon_batch, in_out, reduction='sum') / in_out.size(0)
            
            loss = reconstruction_loss + vq_commitment_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            train_loss += loss.item()
            recon_loss += reconstruction_loss.item()
            vq_loss += vq_commitment_loss.item()
            
            optimizer.step()
            num_batches += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {in_out.shape}")
            continue
    
    avg_loss = train_loss / max(1, num_batches)
    avg_recon_loss = recon_loss / max(1, num_batches)
    avg_vq_loss = vq_loss / max(1, num_batches)
    
    print(f'Epoch: {epoch}, Average training loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, VQ: {avg_vq_loss:.4f}), Batches processed: {num_batches}')
    
    return avg_loss

def validate(model: AbstractVQVAE, val_loader, device, epoch=0):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    for batch_idx, (input, output) in enumerate(val_loader):
        in_out = torch.cat((input, output), dim=0)
        in_out = in_out.to(device)
        
        try:
            recon_batch, vq_commitment_loss, _, _ = model(in_out)
            
            reconstruction_loss = F.binary_cross_entropy(recon_batch, in_out, reduction='sum') / in_out.size(0)
            
            loss = reconstruction_loss + vq_commitment_loss
            
            val_loss += loss.item()
            
            num_batches += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {in_out.shape}")
            continue
    
    avg_loss = val_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average validation loss: {avg_loss:.4f}, Batches processed: {num_batches}')
    return avg_loss