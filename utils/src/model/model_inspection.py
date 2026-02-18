import torch

from model.src.paths import BEST_MODEL_PATH

checkpoint = torch.load(BEST_MODEL_PATH)
epoch = checkpoint['epoch']
model_state_dict = checkpoint['model_state_dict']
optimizer_state_dict = checkpoint['optimizer_state_dict']
val_loss = checkpoint['val_loss']
val_acc = checkpoint['val_acc']

print(f"Trained for {epoch} epochs, reaching validation {val_loss} loss and {val_acc} accuracy.")