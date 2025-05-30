import torch
import torch.nn.functional as F

def contrastive_loss(mu, temperature=0.1):
    batch_size = mu.size(0)
    assert batch_size % 2 == 0

    mu = F.normalize(mu, dim=1) 

    anchors = mu[::2]

    
    similarity = torch.matmul(anchors, mu.T) / temperature  # [B/2, B]

    labels = torch.arange(anchors.size(0), device=mu.device)
    
    loss = F.cross_entropy(similarity, labels)

    return loss


def vae_loss(recon_logits, x, mu, logvar, beta=1.0, contrastive_weight=0.0):
    recon_logits = torch.clamp(recon_logits, 0.0, 1.0)
    target = torch.clamp(x, 0.0, 1.0)

    recon_loss = F.binary_cross_entropy(recon_logits, target, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    contrast_loss = 0.0
    if contrastive_weight > 0:
        contrast_loss = contrastive_loss(mu)
        total_loss += contrastive_weight * contrast_loss

    return total_loss, recon_loss, kl_loss, contrast_loss

def train(model, train_loader, loss_fn, optimizer, device, beta=1.0, contrastive_weight=0.0, epoch=0):
    model.train()
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    contrast_loss_total = 0
    num_batches = 0
    
    for batch_idx, (input, output) in enumerate(train_loader):
        
        in_out = torch.cat((input, output), dim=0).to(device)

        optimizer.zero_grad()
        
        try:
            recon_batch, mu, logvar = model(in_out)
            loss, rl, kl, contrast = loss_fn(recon_batch, in_out, mu, logvar, beta, contrastive_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            recon_loss += rl.item()
            kl_loss += kl.item()
            contrast_loss_total += contrast if isinstance(contrast, float) else contrast.item()
            optimizer.step()
            num_batches += 1
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            print(f"Input shape: {in_out.shape}")
            continue
    
    avg_loss = train_loss / max(1, num_batches)
    avg_recon_loss = recon_loss / max(1, num_batches)
    avg_kl_loss = kl_loss / max(1, num_batches)
    avg_contrast_loss = contrast_loss_total / max(1, num_batches)

    print(f'Epoch: {epoch}, Avg train loss: {avg_loss:.4f} (RL: {avg_recon_loss}, KL: {avg_kl_loss}, CL: {avg_contrast_loss}), Batches: {num_batches}')
    return avg_loss

def validate(model, val_loader, loss_fn, device, beta=1.0, contrastive_weight=0.0, epoch=0):
    model.eval()
    val_loss = 0
    recon_loss = 0
    kl_loss = 0
    contrast_loss_total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (input, output) in enumerate(val_loader):
            in_out = torch.cat((input, output), dim=0).to(device)
            
            try:
                recon_batch, mu, logvar = model(in_out)
                loss, rl, kl, contrast = loss_fn(recon_batch, in_out, mu, logvar, beta, contrastive_weight)
                
                val_loss += loss.item()
                recon_loss += rl.item()
                kl_loss += kl.item()
                contrast_loss_total += contrast if isinstance(contrast, float) else contrast.item()
                num_batches += 1

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                print(f"Input shape: {in_out.shape}")
                continue

    avg_loss = val_loss / max(1, num_batches)
    avg_recon_loss = recon_loss / max(1, num_batches)
    avg_kl_loss = kl_loss / max(1, num_batches)
    avg_contrast_loss = contrast_loss_total / max(1, num_batches)

    print(f'Epoch: {epoch}, Avg validation loss: {avg_loss:.4f} (RL: {avg_recon_loss}, KL: {avg_kl_loss}, CL: {avg_contrast_loss}), Batches: {num_batches}')
    return avg_loss
