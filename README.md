# 生成式对抗网络

## 什么是GAN?

**GAN**是通过**对抗训练**的方式来使得**生成网络产生的样本**服从**真实数据分布**。而其网络的关键在于**生成网络**和**判别网络**的对抗学习。

- **判别网络**，目标是尽量准确地判断一个样本是来自于真实数据还是由生成网络产生；
- **生成网络**，目标是尽量生成判别网络无法区分来源的样本。

这两个目标相反的网络不断地进行交替训练。当最后收敛时，如果判别网络再也无法判断出一个样本的来源，那么也就等价于生成网络可以生成符合真实数据分布的样本。生成对抗网络的流程图如下所示。

![](https://raw.githubusercontent.com/datawhalechina/dive-into-cv-pytorch/master/markdown_imgs/chapter05/GAN.png)

## GAN网络的结构

GAN从网络的角度来看，它由**两部分**组成。

- **生成器网络**：它一个潜在空间的随机向量作为输入，并将其解码为一张合成图像；
- **判别器网络**：以一张图像（真实的或合成的均可）作为输入，并预测该图像来自训练集还是来自生成器网络。

![](https://production-media.paperswithcode.com/methods/gan.jpeg)

## 损失函数

| 符号      | 说明          |
| --------- | ------------- |
| $D(x)$    | Discriminator |
| $G(z)$    | Generator     |
| $x^{(i)}$ | Real Sample   |
| $z^{(i)}$ | Random Noise  |

**Discriminator**: 

$$\displaystyle\dfrac1m\sum_{i=1}^m\left[\log D(x^{(i)})+\log(1-D(G(z^{(i)})))\right],$$

希望准确地判断出真的和假的，追求最大化

**Generator**：

$$\displaystyle\dfrac1m\sum_{i=1}^m\log(1-D(G(z^{(i)}))),$$

希望Discriminator把假的当成真的，追求最小化

把这两部分**加在一起**：

$$\displaystyle\min_G\max_D V(D,G)=\mathbb{E}_{\boldsymbol{x}\sim p_\mathrm{data}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z}\sim p_\mathrm{z}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]$$

实际训练中**Generator**的Loss用的是这个：

$$\displaystyle\max_G\mathbb{E}_{\boldsymbol{z}\sim p_\mathrm{z}(\boldsymbol{z})}[\log D(G(\boldsymbol{z}))],$$

这样的损失函数leads to [non-saturating gradients](https://blog.csdn.net/yzy_1996/article/details/112648606)，训练起来更容易

## Colab Notebook

* [GAN](notebook/gan.ipynb)
* [DCGAN](notebook/dcgan.ipynb)
* [WGAN](notebook/wgan.ipynb)
* [WGAN-GP](notebook/wgan_gp.ipynb)
* [CGAN](notebook/cgan.ipynb)
* [Pix2Pix](notebook/pix2pix.ipynb)