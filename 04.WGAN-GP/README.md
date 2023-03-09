## WGAN-GP

Resources and papers:

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

[Wasserstein GAN最新进展：从weight clipping到gradient penalty，更加先进的Lipschitz限制手法](https://www.zhihu.com/question/52602529/answer/158727900)

判别器loss：

$$L(D)=-\mathbb{E}_{x\sim\mathbb{P}_r}[D(x)]+\mathbb{E}_{x\sim\mathbb{P}_g}[D(x)]+\lambda\mathbb{E}_{x\sim \mathcal{P}_{\hat{x}}}[\|\nabla_xD(x)\|_p-1]^2$$

> * **Penalty coefficient** All experiments in this paper use **λ = 10,** which we found to work well across a variety of architectures and datasets ranging from toy tasks to large ImageNet CNNs.
> * **No critic batch normalization** Most prior GAN implementations use batch normalization in both the generator and the discriminator to help stabilize training, but batch normalization changes the form of the discriminator’s problem from mapping a single input to a single output to mapping from an entire batch of inputs to a batch of outputs. Our penalized training objective is no longer valid in this setting, since we penalize the norm of the critic’s gradient with respect to each input independently, and not the entire batch. To resolve this, we simply omit batch normalization in the critic in our models, finding that they perform well without it. Our method works with normalization schemes which don’t introduce correlations between examples. In particular, we recommend **layer normalization** as a drop-in replacement for batch normalization.

## Gradient Penalty

```python
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty
```

## Problems with Weight Clipping

> Weight clipping is a clearly terrible way to enforce a Lipschitz constraint. If the clipping parameter is large, then it can take a long time for any weights to reach their limit, thereby making it harder to train the critic till optimality. If the clipping is small, this can easily lead to vanishing gradients when the number of layers is big, or batch normalization is not used (such as in RNNs). We experimented with simple variants (such as projecting the weights to a sphere) with little difference, and we stuck with weight clipping due to its simplicity and already good performance.
>
> ![img](https://pic1.zhimg.com/80/v2-27afb895eea82f5392b19ca770865b96_720w.webp?source=1940ef5c)

> Weight clipping (left) pushes weights towards two values (the extremes of the clipping range), unlike gradient penalty (right)
>
> ![img](https://pic1.zhimg.com/80/v2-34114a10c56518d606c1b5dd77f64585_720w.webp?source=1940ef5c)
>
> Gradient norms of deep WGAN critics during training on the Swiss Roll dataset either explode or vanish when using weight clipping, but not when using a gradient penalty.

## Empirical Results

> CIFAR-10 Inception score over generator iterations (left) or wall-clock time (right) for four models: WGAN with weight clipping, WGAN-GP with RMSProp and Adam (to control for the optimizer), and DCGAN. WGAN-GP significantly outperforms weight clipping and performs comparably to DCGAN.
>
> ![img](https://pic1.zhimg.com/80/v2-5b01ef93f60a14e7fa10dbea2b620627_720w.webp?source=1940ef5c)

>  Different GAN architectures trained with different methods. We only succeeded in training every architecture with a shared set of hyperparameters using WGAN-GP.
>
> ![img](https://picx.zhimg.com/80/v2-e0a3d86ccfa101a4d3fee1c0cef96a81_720w.webp?source=1940ef5c)