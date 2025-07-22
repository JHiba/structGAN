

##with PSNR

import os
from PIL import Image
import torch
from torchvision import transforms
from pix import SimpleUNet  # Import your model from your training script
import torchvision.utils as vutils
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
testA_dir = 'C:/Users/graan/structgan/data/test_A'
testB_dir = 'C:/Users/graan/structgan/data/test_B'  # For ground truth, optional
model_path = 'outputs_pix2pix1/generator.pth'
output_dir = 'outputs_pix2pix1/test_fake'
os.makedirs(output_dir, exist_ok=True)

# Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Denormalization helper for saving images
def denorm(tensor):
    tensor = tensor * 0.5 + 0.5  # from [-1,1] to [0,1]
    return tensor.clamp(0,1)

# PSNR calculation (expects tensors in [0,1])
def calculate_psnr(fake, gt):
    mse = torch.mean((fake - gt) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# Load model
netG = SimpleUNet(input_nc=3, output_nc=3, ngf=64).to(device)
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.eval()

# L1 loss for overfitting check
l1 = torch.nn.L1Loss()
total_l1_loss = 0
total_psnr = 0
count = 0

# Run inference on all testA images and compute L1/PSNR if ground truth available
for fname in sorted(os.listdir(testA_dir)):
    imgA_path = os.path.join(testA_dir, fname)
    imgA = Image.open(imgA_path).convert('RGB')
    imgA_tensor = transform(imgA).unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        fake_tensor = netG(imgA_tensor)
        fake_img = denorm(fake_tensor.squeeze(0)).cpu()
        vutils.save_image(fake_img, os.path.join(output_dir, fname))

    # If ground truth exists, calculate L1 and PSNR
    gt_path = os.path.join(testB_dir, fname)
    if os.path.exists(gt_path):
        imgB = Image.open(gt_path).convert('RGB')
        imgB_tensor = transform(imgB).unsqueeze(0).to(device)
        # L1 loss: use tensors before denorm (range [-1, 1])
        loss = l1(fake_tensor, imgB_tensor)
        total_l1_loss += loss.item()

        # PSNR: convert both to [0,1], remove batch dim
        fake_img_01 = denorm(fake_tensor.squeeze(0)).cpu()
        gt_img_01 = denorm(imgB_tensor.squeeze(0)).cpu()
        psnr = calculate_psnr(fake_img_01, gt_img_01)
        total_psnr += psnr
        count += 1

print(f"Generated images are saved in: {output_dir}")

if count > 0:
    print(f"Average L1 loss on test set: {total_l1_loss / count:.4f}")
    print(f"Average PSNR on test set: {total_psnr / count:.2f} dB")
else:
    print("No ground truth images found, so L1/PSNR was not computed.")
