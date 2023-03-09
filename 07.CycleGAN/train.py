import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import HorseZebraDataset
from discriminator_model import Discriminator
from generator_model import Generator
from utils import save_checkpoint, load_checkpoint


def train_fn(disc_h, disc_z, gen_z, gen_h, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_horse = gen_h(zebra)
            d_h_real = disc_h(horse)
            d_h_fake = disc_h(fake_horse.detach())
            d_h_real_loss = mse(d_h_real, torch.ones_like(d_h_real))
            d_h_fake_loss = mse(d_h_fake, torch.zeros_like(d_h_fake))
            d_h_loss = d_h_real_loss + d_h_fake_loss

            fake_zebra = gen_z(horse)
            d_z_real = disc_z(zebra)
            d_z_fake = disc_z(fake_zebra.detach())
            d_z_real_loss = mse(d_z_real, torch.ones_like(d_z_real))
            d_z_fake_loss = mse(d_z_fake, torch.zeros_like(d_z_fake))
            d_z_loss = d_z_real_loss + d_z_fake_loss

            # put it together
            d_loss = (d_h_loss + d_z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            d_h_fake = disc_h(fake_horse)
            d_z_fake = disc_z(fake_zebra)
            loss_g_h = mse(d_h_fake, torch.ones_like(d_h_fake))
            loss_g_z = mse(d_z_fake, torch.ones_like(d_z_fake))

            # cycle loss
            cycle_zebra = gen_z(fake_horse)
            cycle_horse = gen_h(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss
            identity_zebra = gen_z(zebra)
            identity_horse = gen_h(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)

            # add all together
            g_loss = (
                    loss_g_z
                    + loss_g_h
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")


def main():
    disc_h = Discriminator(in_channels=3).to(config.DEVICE)
    disc_z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_h = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_h.parameters()) + list(disc_z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_z.parameters()) + list(gen_h.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H, gen_h, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z, gen_z, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H, disc_h, opt_disc, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z, disc_z, opt_disc, config.LEARNING_RATE)

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/horses",
        root_zebra=config.TRAIN_DIR + "/zebras",
        transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_h, disc_z, gen_z, gen_h, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler)

        if config.SAVE_MODEL:
            save_checkpoint(gen_h, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_h, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == "__main__":
    main()
