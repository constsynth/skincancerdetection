from torch import nn
import torch
from torchvision import transforms, datasets
from torchvision import models
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
from PIL import Image
import torch.nn.functional as F
from cspackage.image_processing import crop_center, super_resolution
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'datasets/data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}")


model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load('./models/CNN_CAN_V1.pt'))
model_ft = model_ft.to(device)



def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def model_predict(model,img_path):
    was_training = model.training
    model.eval()
    img = Image.open(img_path)
    if img.width > 512 or img.height > 512:
        img = crop_center(img, frac=0.55)
    img = super_resolution(img)
    img = data_transforms['test'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        # print(outputs)
        _, preds = torch.max(outputs, 1)
        probs = F.softmax(outputs, dim=1)
        # print(probs[0])
        # ax = plt.subplot(2,2,1)
        # ax.axis('off')
        # ax.set_title(f'Predicted: {class_names[preds[0]]}')
        # imshow(img.cpu().data[0])

        model.train(mode=was_training)
    return round(probs[0][1].tolist(), 2)


# Get a batch of training data
if __name__ == '__main__':
    # model = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=10)
    result = model_predict(
        model_ft,
        img_path='_92607945_melanoma.jpg'
    )
    print(result)
    # plt.ioff()
    # plt.show()
