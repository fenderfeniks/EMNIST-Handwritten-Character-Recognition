import os
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass, shift
from torchvision import transforms
import cv2 as cv

class CNN(nn.Module):
    def __init__(self, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 5 * 5, n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class Model:
    def __init__(self):
        # загрузка маппинга
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_path = os.path.join(base_dir, '..', 'emnist-balanced-mapping.txt')
        self.mapping = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                label, ascii_code = map(int, line.strip().split())
                self.mapping[label] = chr(ascii_code)

        # загрузка модели
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN(n_classes=len(self.mapping)).to(self.device)
        model_path = os.path.join('myapp', 'model.ckpt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # трансформации
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        x = x.reshape(28, 28).T.astype(np.uint8)

        # утолщаем линии
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        x = cv.dilate(x, kernel, iterations=1)

        # bilateral вместо gaussian — сохраняет края символа
        x = cv.bilateralFilter(x, d=3, sigmaColor=75, sigmaSpace=75)

        if x.max() > 0:
            cy, cx = center_of_mass(x.astype(np.float64))
            if not (np.isnan(cy) or np.isnan(cx)):
                shift_y = 14 - cy
                shift_x = 14 - cx
                x = shift(x.astype(np.float64), [shift_y, shift_x])

        x = (x / 255.0).astype(np.float32)
        x = (x - 0.1307) / 0.3081
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            pred = output.argmax(dim=1).item()

        return self.mapping[pred]