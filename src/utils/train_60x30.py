import torch

def train(model, train_loader, loss_fn, optimizer, device, beta=1.0, epoch=0):
    model.train()
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    num_batches = 0
    
    for batch_idx, (input, output) in enumerate(train_loader):
        
        pair = torch.cat((input, output), dim=3) # make them 30x60
        pair_reverse = torch.cat((output, input), dim=3) # make them 30x60
        
        in_out = torch.cat((pair, pair_reverse), dim=0).to(device)

        optimizer.zero_grad()
        
        try:
            recon_batch, mu, logvar = model(in_out)
            loss, rl, kl = loss_fn(recon_batch, in_out, mu, logvar, beta)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            recon_loss += rl.item()
            kl_loss += kl.item()
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

def validate(model, val_loader, loss_fn, device, beta=1.0, epoch=0):
    model.eval()
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (input, output) in enumerate(val_loader):
            in_out = torch.cat((input, output), dim=3) # make them 30x60
            in_out = in_out.to(device)
            
            try:
                recon_batch, mu, logvar = model(in_out)
                loss, _, _ = loss_fn(recon_batch, in_out, mu, logvar, beta)
                val_loss += loss.item()
            
                num_batches += 1
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                print(f"Input shape: {in_out.shape}")
                continue
    
    avg_loss = val_loss / max(1, num_batches)
    print(f'Epoch: {epoch}, Average validation loss: {avg_loss:.4f}, Batches processed: {num_batches}')
    return avg_loss
