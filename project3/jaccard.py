import torch

def jaccard_loss(true, logits, eps=1e-7):
    """Computes the Jaccard loss, a.k.a IoU loss.
    
    Note that PyTorch optimizers minimize a loss. In this case, you want to minimize the complement
    of the IoU to make the loss function work.
    
    Args:
        true: a tensor of ground truth values
        logits: a tensor of logits
        eps: a small epsilon for numerical stability to avoid division by zero error
    """
    # You may need to apply a sigmoid or softmax to the logits if they are not normalized
    probs = torch.sigmoid(logits)
    true = true.float() # Ensure float precision for true labels
    preds = (probs > 0.5).float() # Binarize predictions to 0 and 1
    
    intersection = (preds * true).sum(dim=(1, 2, 3)) # Sum over the batch, height, width dimensions (in case of 2D images)
    union = (preds + true).sum(dim=(1, 2, 3)) - intersection
    
    iou = (intersection + eps) / (union + eps) # IoU score
    loss = 1 - iou # Subtract from 1 because we want to minimize the loss (maximize IoU)
    return loss.mean() # Return the mean Jaccard loss

# ... in your training loop ...
for epoch in range(num_epochs):
    total_loss = 0
    for (img, mask) in tqdm(loader):
        img = img.to(device=torch.device('cuda'), dtype=torch.float32, non_blocking=True)
        mask = mask.to(device=torch.device('cuda'), dtype=torch.float32, non_blocking=True).unsqueeze(1)
        pred = model(img)
        loss = jaccard_loss(mask, pred)
        # ... the rest of the training loop ...
