# Kanji-Classification-CNN
A CNN model pretrained to label pictures of arbitrary size to japanese kanji characters implemented in PyTorch.

### Dataset
- Source (CLIPCJK_Gen)
- License
- Data augmentation

### Model
- PyTorch
- CNN
- Classification

Using device: cpu
Loading processed data...
Loaded 5000 samples with 1000 classes
Input shape: (5000, 64, 64, 1)
Preparing data loaders...
Train batches: 7032, Val batches: 8
Creating model...
Total parameters: 1,818,184
Trainable parameters: 1,818,184

Starting training for 50 epochs...
--------------------------------------------------------------------------------
Epoch 1, Progress: 100.0%
Epoch 1/50 | Train Loss: 4.5855 | Train Acc: 0.1060 | Val Loss: 1.8944 | Val Acc: 0.4960 | LR: 0.000999
  -> Saved best model (val_loss: 1.8944)
Epoch 2, Progress: 100.0%
Epoch 2/50 | Train Loss: 1.3921 | Train Acc: 0.5935 | Val Loss: 0.6552 | Val Acc: 0.8620 | LR: 0.000995
  -> Saved best model (val_loss: 0.6552)
Epoch 3, Progress: 100.0%
Epoch 3/50 | Train Loss: 0.4546 | Train Acc: 0.8598 | Val Loss: 0.5362 | Val Acc: 0.9220 | LR: 0.000989
  -> Saved best model (val_loss: 0.5362)
Epoch 4, Progress: 100.0%
Epoch 4/50 | Train Loss: 0.2328 | Train Acc: 0.9289 | Val Loss: 0.4506 | Val Acc: 0.9400 | LR: 0.000981
  -> Saved best model (val_loss: 0.4506)
Epoch 5, Progress: 100.0%
Epoch 5/50 | Train Loss: 0.1624 | Train Acc: 0.9517 | Val Loss: 0.4546 | Val Acc: 0.9580 | LR: 0.000970
Epoch 6, Progress: 100.0%
Epoch 6/50 | Train Loss: 0.1275 | Train Acc: 0.9629 | Val Loss: 0.4784 | Val Acc: 0.9560 | LR: 0.000957
Epoch 7, Progress: 12.6%

### Application
- Send screenshot to evaluate
- Read result + confidence level

### Evaluation
- Accuracy during training + validation
- 
