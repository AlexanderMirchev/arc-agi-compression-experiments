import torch
import torch.nn.functional as F

def vae_loss(recon_logits, x, mu, logvar, beta=1.0):
    target = torch.argmax(x, dim=1) 
    recon_loss = F.cross_entropy(recon_logits, target, reduction='mean') 
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss

def train(model, train_loader, optimizer, device, beta=1.0, epoch=0):
    model.train()
    train_loss = 0
    num_batches = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data[0].to(device) if isinstance(data, list) else data.to(device)
        optimizer.zero_grad()
        
        try:
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar, beta)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            num_batches += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {data.shape}")
            continue
    
    avg_loss = train_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average training loss: {avg_loss:.4f}, Batches processed: {num_batches}')
    return avg_loss

def validate(model, val_loader, device, beta=1.0, epoch=0):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            data = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            try:
                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar, beta)
                val_loss += loss.item()
            
                num_batches += 1
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                print(f"Input shape: {data.shape}")
                continue
    
    avg_loss = val_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average validation loss: {avg_loss:.4f}, Batches processed: {num_batches}')
    return avg_loss
