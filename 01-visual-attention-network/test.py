# Prepare CIFAR 10000 Images
import torchvision
import torchvision.transforms as transforms
from .model import van_tiny

dataset = torchvision.datasets.CIFAR10(root="data", train=False, download=True)
model = van_tiny(pretrained=True).cuda()
img_idx = 0
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
for idx, img in enumerate(dataset):
    if img_idx == 16:
        break
    with torch.no_grad():
        img_th = transform(img[0]).unsqueeze(0).cuda()
        output = model(img_th)
        print(f'[{idx}] output : {output.shape}')
    img_idx += 1
