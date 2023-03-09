import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MapDataset
from discriminator_model import Discriminator
from generator_model import Generator
from utils import save_checkpoint, load_checkpoint, save_some_examples

torch.backends.cudnn.benchmark = True


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            d_real = disc(x, y)
            d_fake = disc(x, y_fake.detach())
            d_real_loss = bce(d_real, torch.ones_like(d_real))
            d_fake_loss = bce(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            d_fake = disc(x, y_fake)
            g_fake_loss = bce(d_fake, torch.ones_like(d_fake))
            l1_loss = l1(y_fake, y) * config.L1_LAMBDA
            g_loss = g_fake_loss + l1_loss

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir="data/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir="data/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__ == '__main__':
    main()
