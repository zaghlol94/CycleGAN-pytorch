import torch
from utils import load_checkpoint
import torch.optim as optim
import config
from torchvision.utils import save_image
from generator import Generator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="cycle gan for horses<->zeras and vice-versa.")
parser.add_argument("-i", "--image", type=str, required=True, help="origin image")
parser.add_argument("-s", "--start", type=str, required=True,
                    help="this argument should be the starting point of the image either horse if you pass "
                         "horse photo or zebra if you pass horse ")
args = parser.parse_args()

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
)

gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
opt_gen = optim.Adam(
    list(gen_Z.parameters()) + list(gen_H.parameters()),
    lr=config.LEARNING_RATE,
    betas=(0.5, 0.999),
)

load_checkpoint(
    config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
)
load_checkpoint(
    config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
)

with torch.no_grad():
    if args.start == "horse":
        horse_img = np.array(Image.open(args.image).convert("RGB"))
        augmentations = transforms(image=horse_img)
        horse_img = augmentations["image"]
        horse_img = horse_img.unsqueeze(0).to(config.DEVICE)
        fake_zebra = gen_Z(horse_img)
        save_image(fake_zebra * 0.5 + 0.5, "zebra_fake.png")
        reconstruct = gen_H(fake_zebra)
        save_image(reconstruct * 0.5 + 0.5, "H_reconstructed_fake.png")
    ######################################################
    elif args.start == "zebra":
        zebra_img = np.array(Image.open(args.image).convert("RGB"))
        augmentations = transforms(image=zebra_img)
        zebra_img = augmentations["image"]
        zebra_img = zebra_img.unsqueeze(0).to(config.DEVICE)
        fake_horse = gen_H(zebra_img)
        save_image(fake_horse * 0.5 + 0.5, "horse_fake.png")
        reconstruct = gen_Z(fake_horse)
        save_image(reconstruct * 0.5 + 0.5, "Z_reconstructed_fake.png")
    else:
        print("starting point should be either horse or zebra ")
