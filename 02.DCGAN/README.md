## DCGAN

Resources and papers:

[Unsupervised Representation Learning with Deep Conventional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-01_at_11.27.51_PM_IoGbo1i.png)

> Architecture guidelines for stable Deep Convolutional GANs
>
> * Replace any pooling layers with **strided convolutions** (discriminator) and **fractional-strided convolutions** (generator).
> * Use **batchnorm** in both the generator and the discriminator.
> * **Remove fully connected hidden layers** for deeper architectures.
> * Use **ReLU** activation in generator for all layers except for the output, which uses **Tanh**.
> * Use **LeakyReLU** activation in the discriminator for all layers.

> Details of adversarial training
>
> * No pre-processing was applied to training images besides scaling to the range of the **tanh** activation function [-1, 1].
> * All models were trained with mini-batch stochastic gradient descent (SGD) with **a mini-batch size of 128**.
> * All weights were initialized from a **zero-centered** Normal distribution with **standard deviation 0.02**.
> * In the **LeakyReLU**, the slope of the leak was set to **0.2** in all models.
> * While previous GAN work has used momentum to accelerate training, we used the Adam optimizer (Kingma & Ba, 2014) with tuned hyperparameters. We found the suggested **learning rate** of 0.001, to be too high, using **0.0002** instead. Additionally, we found leaving the momentum term **β1** at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to **0.5** helped stabilize training.

## Network

### Discriminator

| Letter | Meaning             |
| ------ | ------------------- |
| $I$    | Input: $I\times I$  |
| $K$    | kernel_size: $K$    |
| $P$    | padding: $P$        |
| $S$    | stride: $S$         |
| $O$    | Output: $O\times O$ |

$O=\dfrac{I+2P-K}{S}+1$

```python
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # 32 x 32
            # According to PyTorch-GAN, no batchnorm here
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),  # 16 x 16
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),  # 8 x 8
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),  # 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),  # 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)
```

### Generator

| Letter | Meaning             |
| ------ | ------------------- |
| $I$    | Input: $I\times I$  |
| $K$    | kernel_size: $K$    |
| $P$    | padding: $P$        |
| $S$    | stride: $S$         |
| $O$    | Output: $O\times O$ |

$O=S(I-1)-2P+K+\lfloor\dfrac{O+2P-K}{S}\rfloor$

```python
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 1 x 1
            self._block(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0),  # 4 x 4
            self._block(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),  # 8 x 8
            self._block(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),  # 16 x 16
            self._block(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),  # 32 x 32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),  # 64 x 64
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False,),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)
```

### Initialize

```python
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.real, 0.0, 0.02)
```