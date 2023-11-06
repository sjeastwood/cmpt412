import torch

def dice_loss(pred, target, smooth = 1.):
    pred = torch.sigmoid(pred)  # Apply sigmoid to get the probability for each pixel
    # Flatten the prediction and target tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()  # Intersection part of the Dice coefficient
    union = pred_flat.sum() + target_flat.sum()  # Union part of the Dice coefficient
    
    dice_score = (2. * intersection + smooth) / (union + smooth)  # Dice score
    dice_loss = 1 - dice_score  # Dice loss
    
    return dice_loss

# Now, replace BCEWithLogitsLoss with dice_loss in your training loop
num_epochs = 5
batch_size = 4
learning_rate = 0.01
weight_decay = 1e-5

model = MyModel() # initialize the model
model = model.cuda() # move the model to GPU
loader, _ = get_plane_dataset('train', batch_size) # initialize data_loader
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Initialize the optimizer as SGD

for epoch in range(num_epochs):
    total_loss = 0
    for (img, mask) in tqdm(loader):
        img = img.cuda()  # Assuming that img and mask are already tensors
        mask = mask.cuda().unsqueeze(1)  # Adding channel dimension to mask if necessary
        pred = model(img)
        loss = dice_loss(pred, mask)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print("Epoch: {}, Loss: {}".format(epoch, avg_loss))
    torch.save(model.state_dict(), '{}/output/{}_segmentation_model.pth'.format(BASE_DIR, epoch))

# Saving the final model
torch.save(model.state_dict(), '{}/output/final_segmentation_model.pth'.format(BASE_DIR))
