import torch.cuda
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from tqdm.contrib import tenumerate

from model import Discriminator, Generator

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-4  # 3e-4 seems the better learning rate for Adam
Z_DIM = 64  # also you can try 128, 256
IMAGE_DIM = 784  # 28 * 28 * 1
BATCH_SIZE = 32
NUM_EPOCHS = 50

transforms = transforms.Compose(
    [
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range[0, 255]
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.ToTensor(),
        # normalize to N(mean, std)
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
disc = Discriminator(IMAGE_DIM).to(device)
gen = Generator(Z_DIM, IMAGE_DIM).to(device)

opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

fixed_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in tenumerate(dataloader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        noise = torch.randn(batch_size, Z_DIM).to(device)
        fake = gen(noise)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = make_grid(fake, normalize=True)
                img_grid_real = make_grid(data, normalize=True)

                writer_real.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
