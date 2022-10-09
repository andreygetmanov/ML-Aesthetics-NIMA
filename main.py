import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json


def rate(img_path):
    """
    Returns: Scores, mean, std
    """
    # Number of classes in the dataset
    num_classes = 10

    model_ft = models.densenet121(pretrained=True)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Sequential(
        nn.Linear(num_ftrs,num_classes),
        nn.Softmax(1)
    )

    # Weight Path
    weight_path = 'weights/dense121_all.pt'

    # Load weights
    assert os.path.exists(weight_path)
    model_ft.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

    scores_dict = {}
    for image in os.listdir(img_path):
        img = Image.open(os.path.join(img_path, image))
        transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        img = transform(img)

        with torch.no_grad():
            scores = model_ft(img.view(1,3,224,224))
            weighted_votes = torch.arange(10, dtype=torch.float) + 1
            mean = torch.matmul(scores, weighted_votes)
            std = torch.sqrt((scores * torch.pow((weighted_votes - mean.view(-1,1)), 2)).sum(dim=1))
        print(image, np.round(mean.item(), 3), np.round(std.item(), 3))
        scores_dict[image] = mean.item()
    return scores_dict


if __name__ == '__main__':

    scores_dict = rate('images')
    with open("scores.json", "w") as outfile:
        json.dump(scores_dict, outfile)
    print(outfile)