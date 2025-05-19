import numpy as np
import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, device, beta=1.0, epoch=0):
    model.train()
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    num_batches = 0
    
    for batch_idx, (input, output) in enumerate(train_loader):
        
        in_out = torch.cat((input, output), dim=0)
        # in_out = input
        in_out = in_out.to(device)

        optimizer.zero_grad()
        
        try:
            mu, logvar = model.encode(in_out)
            z = model.reparameterize(mu, logvar)
            recon_batch = model.decode(z)

            rl = F.binary_cross_entropy(recon_batch, in_out, reduction='sum') / in_out.size(0)

            q_log_posterior = -0.5 * torch.sum(
                np.log(2 * np.pi) + logvar + ((z - mu) ** 2) / torch.exp(logvar), dim=1
            )

            p_log_prior = model.compute_vampprior_logp(z, device)

            kl_div = q_log_posterior - p_log_prior
            kl_mean = kl_div.mean()

            loss = rl + beta * kl_mean
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            recon_loss += rl.item()
            kl_loss += kl_mean.item()
            optimizer.step()
            num_batches += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {in_out.shape}")
            continue
    
    avg_loss = train_loss / max(1, num_batches)
    avg_recon_loss = recon_loss / max(1, num_batches)
    avg_kl_loss = kl_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average training loss: {avg_loss:.4f} (RL: {avg_recon_loss}, KL: {avg_kl_loss}), Batches processed: {num_batches}')
    return avg_loss

def validate(model, train_loader, device, beta=1.0, epoch=0):
    model.eval()
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    num_batches = 0
    
    for batch_idx, (input, output) in enumerate(train_loader):
        
        in_out = torch.cat((input, output), dim=0)
        # in_out = input
        in_out = in_out.to(device)

        try:
            mu, logvar = model.encode(in_out)
            z = model.reparameterize(mu, logvar)
            recon_batch = model.decode(z)

            rl = F.binary_cross_entropy(recon_batch, in_out, reduction='sum') / in_out.size(0)

            q_log_posterior = -0.5 * torch.sum(
                np.log(2 * np.pi) + logvar + ((z - mu) ** 2) / torch.exp(logvar), dim=1
            )

            p_log_prior = model.compute_vampprior_logp(z, device)

            kl_div = q_log_posterior - p_log_prior
            kl_mean = kl_div.mean()

            loss = rl + beta * kl_mean
            
            train_loss += loss.item()
            recon_loss += rl.item()
            kl_loss += kl_mean.item()
            num_batches += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {in_out.shape}")
            continue 
    
    avg_loss = train_loss / max(1, num_batches)
    avg_recon_loss = recon_loss / max(1, num_batches)
    avg_kl_loss = kl_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average validation loss: {avg_loss:.4f} (RL: {avg_recon_loss}, KL: {avg_kl_loss}), Batches processed: {num_batches}')
    return avg_loss
