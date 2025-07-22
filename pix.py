import os
from PIL import Image
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
from hashlib import md5
import torch.nn.functional as F
import math

def calculate_psnr(fake, real, max_val=1.0):
    # Assumes [B, C, H, W] and range [0, 1]
    mse = F.mse_loss(fake, real)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr

# 1. Dataset (Paired)

class PairedDataset(Dataset):
    def __init__(self, dir_A, dir_B, transform=None):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transform = transform

        files = sorted(os.listdir(dir_A))
        total = 0
        unique_hashes = set()
        unique_files = []
        for f in files:
            pathA = os.path.join(dir_A, f)
            pathB = os.path.join(dir_B, f)
            if not (os.path.isfile(pathA) and os.path.isfile(pathB)):
                continue
            total += 1
            try:
                with open(pathA, 'rb') as fa, open(pathB, 'rb') as fb:
                    hA = md5(fa.read()).hexdigest()
                    hB = md5(fb.read()).hexdigest()
                    hash_pair = (hA, hB)
                if hash_pair not in unique_hashes:
                    unique_hashes.add(hash_pair)
                    unique_files.append(f)
            except Exception as e:
                print(f"Skipping file {f} due to error: {e}")

        print(f"Total images found: {total}")
        print(f"Images after removing duplicates: {len(unique_files)}")
        self.files = unique_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_A = Image.open(os.path.join(self.dir_A, self.files[idx])).convert('RGB')
        img_B = Image.open(os.path.join(self.dir_B, self.files[idx])).convert('RGB')
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B

# 2. U-Net Generator
class SimpleUNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True)
        )  # 256 -> 128
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, True)
        )  # 128 -> 64
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, True)
        )  # 64 -> 32
        self.down4 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True)
        )  # 32 -> 16
        self.down5 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, True)
        )  # 16 -> 8
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True)
        )  # 8 -> 16
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*8*2, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )  # 16 -> 32
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4*2, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )  # 32 -> 64
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )  # 64 -> 128
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, output_nc, 4, 2, 1),
            nn.Tanh()
        )  # 128 -> 256

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        # Decoder with skip connections
        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))
        u5 = self.up5(torch.cat([u4, d1], 1))
        return u5

# 3. PatchGAN Discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ndf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, 4, 2, 1),  # 256->128
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1),                # 128->64
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),              # 64->32
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1),              # 32->31
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, 1, 1),                  # 31->30
            nn.Sigmoid()
        )

    def forward(self, input_img, output_img):
        # Concatenate along channel dim (B, 6, H, W)
        x = torch.cat([input_img, output_img], dim=1)
        return self.model(x)

# 4. Training Loop
def train(
    data_dirA='C:/Users/graan/structgan/data/train_A',
    data_dirB='C:/Users/graan/structgan/data/train_B',
    epochs=100,
    batch_size=4,
    lr=2e-4,
    out_dir='outputs_pix2pix1'
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    dataset = PairedDataset(data_dirA, data_dirB, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")

    netG = SimpleUNet(input_nc=3, output_nc=3, ngf=64).to(device)
    netD = PatchDiscriminator(input_nc=3, output_nc=3, ndf=64).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    criterionGAN = nn.BCELoss()
    criterionL1 = nn.L1Loss()

    for epoch in range(epochs):
        netG.train()
        netD.train()
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_l1_loss = 0.0
        running_psnr = 0.0
        
        for i, (input_img, target_img) in enumerate(dataloader):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            batch_size_ = input_img.size(0)

            # PatchGAN output shape: (batch, 1, 30, 30) for 256x256 images
            real_label = torch.ones((batch_size_, 1, 30, 30), device=device)
            fake_label = torch.zeros((batch_size_, 1, 30, 30), device=device)

            # ---- Train Discriminator ----
            optimizerD.zero_grad()
            output_real = netD(input_img, target_img)
            loss_D_real = criterionGAN(output_real, real_label)
            fake_img = netG(input_img).detach()
            output_fake = netD(input_img, fake_img)
            loss_D_fake = criterionGAN(output_fake, fake_label)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()

            # ---- Train Generator ----
            optimizerG.zero_grad()
            fake_img = netG(input_img)
            output_fake = netD(input_img, fake_img)
            loss_G_GAN = criterionGAN(output_fake, real_label)
            loss_G_L1 = criterionL1(fake_img, target_img)
            lambda_L1 = 100
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
            loss_G.backward()
            optimizerG.step()


            # Compute PSNR (denormalize for correct range)
            fake_img_denorm = fake_img * 0.5 + 0.5
            target_img_denorm = target_img * 0.5 + 0.5
            psnr_val = calculate_psnr(fake_img_denorm, target_img_denorm)
            running_psnr += psnr_val

            running_g_loss += loss_G.item()
            running_d_loss += loss_D.item()
            running_l1_loss += loss_G_L1.item()
            

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                      f"G Loss: {loss_G.item():.4f}, D Loss: {loss_D.item():.4f}")

        avg_g_loss = running_g_loss / len(dataloader)
        avg_d_loss = running_d_loss / len(dataloader)
        avg_l1_loss = running_l1_loss / len(dataloader)
        avg_psnr = running_psnr / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | L1 Loss: {avg_l1_loss:.4f} | PSNR: {avg_psnr:.2f}")

        # Save sample outputs
        if (epoch+1) % 10 == 0 or epoch == 0:
            netG.eval()
            with torch.no_grad():
                fake = netG(input_img)
                fake = fake * 0.5 + 0.5
                target_img_ = target_img * 0.5 + 0.5
                input_img_ = input_img * 0.5 + 0.5
                vutils.save_image(fake, f"{out_dir}/fake_epoch{epoch+1}.png", normalize=False)
                vutils.save_image(target_img_, f"{out_dir}/real_epoch{epoch+1}.png", normalize=False)
                vutils.save_image(input_img_, f"{out_dir}/input_epoch{epoch+1}.png", normalize=False)

    # Save models
    torch.save(netG.state_dict(), os.path.join(out_dir, "generator.pth"))
    torch.save(netD.state_dict(), os.path.join(out_dir, "discriminator.pth"))
    print(f"Models saved to {out_dir}")

if __name__ == "__main__":
    train(
        data_dirA='C:/Users/graan/structgan/data/train_A',
        data_dirB='C:/Users/graan/structgan/data/train_B',
        epochs=50,
        batch_size=4,
        lr=2e-4,
        out_dir='outputs_pix2pix1'
    )
