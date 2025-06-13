import os, glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CachedLandmarkDataset(Dataset):
    def __init__(self, cache_dir):
        # find all landmark files (tried using regex but glob was simpler)
        self.files = glob.glob(os.path.join(cache_dir, "*.npz"))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        # load cached landmark data - much faster than processing video each time after i tried
        data = np.load(self.files[idx])
        seq = torch.from_numpy(data["seq"]).float()  
        label = int(data["label"])
        return seq, label

class SignLanguageModel(nn.Module):
    def __init__(self, input_size=126, hidden_size=256, num_layers=2, num_classes=2000, dropout=0.5):
        super().__init__()
        # using gru instead of lstm - tried both but gru seemed to work better
        # 126 input = 21 landmarks x 3 coords x 2 hands
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # get hidden states from all the timesteps (batch_size, seq_len, hidden_size)
        h, _ = self.gru(x)
        # only use the final hidden state for classification
        return self.fc(h[:,-1,:])

def train(model, loader, epochs=10, lr=1e-3, ckpt="best.pt"):
    # mixed precision training to speed up the training time
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # tried sgd first but adam converged faster
    opt = torch.optim.Adam(model.parameters(), lr=lr) 
    
    # reduce lr when we plateau
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)
    crit = nn.CrossEntropyLoss()
    best_acc = 0
    
    for e in range(epochs):
        model.train()
        total = correct = 0
        running_loss = 0.0
        
        # might be faster with tqdm progress bar but this works for now
        for i, (seq, lab) in enumerate(loader):
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            opt.zero_grad()
            
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    out = model(seq)
                    loss = crit(out, lab)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out = model(seq)
                loss = crit(out, lab)
                loss.backward()
                opt.step()
            
            running_loss += loss.item()
            pred = out.argmax(1)
            total += lab.size(0)
            correct += (pred==lab).sum().item()
            
            if i % 20 == 19:  # print every 20 mini-batches
                # print(f"[{e+1}, {i+1}] loss: {running_loss/20:.3f}")
                running_loss = 0.0
        
        acc = correct/total
        print(f"Epoch {e+1}/{epochs}  acc={acc:.4f}  lr={opt.param_groups[0]['lr']:.1e}")
        scheduler.step(acc)
        
        # save best model
        if acc > best_acc:
            best_acc = acc
            print(f"Saving best model with accuracy {acc:.4f}")
            torch.save(model.state_dict(), ckpt)

if __name__=="__main__":
    # find fastest algorithms
    torch.backends.cudnn.benchmark = True  
    CACHE_DIR = "/Users/tedgoh/buildingblocs_I27_ml/landmarks_cache"
    
    print("Loading dataset from cached landmarks...")
    dataset = CachedLandmarkDataset(CACHE_DIR)
    print(f"Found {len(dataset)} samples")

    loader = DataLoader(dataset,
                       batch_size=64,
                       shuffle=True,  
                       num_workers=8, 
                       pin_memory=True)  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    
    model = SignLanguageModel(num_classes=2000).to(device)
    # 50 epochs is dogshit, 80 got arnd ~50% acc
    train(model, loader, epochs=80, lr=3e-3, ckpt="best_signlang_model.pt")
