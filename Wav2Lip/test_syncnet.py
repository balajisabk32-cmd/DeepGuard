import torch
from models import SyncNet_color

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SyncNet_color().to(device)
checkpoint_path = 'checkpoints/lipsync_expert.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print("SyncNet loaded successfully!")
print(model)
