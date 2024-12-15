import numpy as np
import torch
!pip install torchvision
from torchvision.models import inceptionv3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from scipy.linalg import sqrtm

def preprocess_image(image, image_size=299):
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def calculate_statistics(features):
    mean = np.mean(features, axis=0)
    covariance = np.cov(features, rowvar=False)
    return mean, covariance

def calculate_fid(mean1, cov1, mean2, cov2):
    mean_diff = np.sum((mean1 - mean2) ** 2)
    cov_sqrt,  = sqrtm(cov1.dot(cov2), disp=False)

    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid_score = mean_diff + np.trace(cov1 + cov2 - 2 * cov_sqrt)
    return fid_score

def extract_features(image, model):
    with torch.no_grad():
        return model(image).flatten(start_dim=1).cpu().numpy()

def calculate_fid_score(image1, image2):
    model = inception_v3(pretrained=True, transform_input=False).eval()
    model.fc = torch.nn.Identity()  # Remove final classification layer

    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    features1 = extract_features(image1, model)
    features2 = extract_features(image2, model)

    mean1, cov1 = calculate_statistics(features1)
    mean2, cov2 = calculate_statistics(features2)

    fid_score = calculate_fid(mean1, cov1, mean2, cov2)
    return fid_score

from PIL import Image

image1 = init_image
image2 = image

fid = calculate_fid_score(image1, image2)
print(f"FID score: {fid}")
