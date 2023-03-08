## WGAN

Resources and papers:

[Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)

[Wasserstein GAN](https://arxiv.org/abs/1701.07875)

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)

[令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913?utm_id=0)

[Wasserstein GAN最新进展：从weight clipping到gradient penalty，更加先进的Lipschitz限制手法](https://www.zhihu.com/question/52602529/answer/158727900)

## What was done

1. 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
2. 基本解决了collapse mode的问题，确保了生成样本的多样性
3. 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高
4. 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到

## The improvement

1. 判别器最后一层去掉sigmoid
   * 原始GAN的判别器做的是二分类任务，所以最后一层是sigmoid，但是现在WGAN中的判别器$f_w$做的是近似拟合Wasserstein距离，属于回归任务，所以要把最后一层的sigmoid拿掉。
2. 生成器和判别器的loss不取log
   * 生成器loss：$-\mathbb{E}_{x\sim\mathbb{P}_g}[f_w(x)]$
   * 判别器loss：$\mathbb{E}_{x\sim\mathbb{P}_g}[f_w(x)]-\mathbb{E}_{x\sim\mathbb{P}_r}[f_w(x)]$
3. 每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c
   * 我们其实不关心具体的$K$是多少，只要它不是正无穷就行，因为它只是会使得梯度变大$K$倍，并不会影响梯度的方向。所以作者采取了一个非常简单的做法，就是限制神经网络$f_\theta$的所有参数$w_i$的不超过某个范围$[−c,\,c]$，比如$w_i\in[−0.01,\,0.01]$，此时关于输入样本$x$的导数$\frac{\part f_w}{\part x}$也不会超过某个范围，所以一定存在某个不知道的常数$K$使得$f_w$的局部变动幅度不会超过它，Lipschitz连续条件得以满足。具体在算法实现中，只需要每次更新完$w$后把它clip回这个范围就可以了。
4. 不要用基于动量的优化算法（包括momentum和Adam），推荐RMSProp，SGD也行
   * 实验发现如果使用Adam，判别器的loss有时候会崩掉。当它崩掉时，Adam给出的更新方向与梯度方向夹角的cos值就变成负数，更新方向与梯度方向南辕北辙，这意味着判别器的loss梯度是不稳定的，所以不适合用Adam这类基于动量的优化算法。作者改用RMSProp之后，问题就解决了，因为RMSProp适合梯度不稳定的情况。

> All experiments in the paper used the default values $\alpha = 0.00005$, $c = 0.01$, $m = 64$, $n_{critic} = 5$. ($n_{critic} = 5$ means $5$ Critic's iterations per Generator's iteration.)

## Network

### Critic

```python
class Critic(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Critic, self).__init__()
        self.disc = nn.Sequential(
            # 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),  # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),  # 16 x 16
            self._block(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),  # 8 x 8
            self._block(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),  # 4 x 4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)  # 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),  # seems LayerNorms <-> InstanceNorm
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)
```

### Generator

```python
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
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
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
```

## Read-through: Wasserstein GAN

### Introduction

When learning generative models, we assume the data have comes from some unknown distribution $P_r$. (The r stands for real.) We want to learn a distribution $P_\theta$ that approximates $P_r$, where $\theta$ are the parameters of the distribution

You can imagine two approaches for doing this.

* Directly learn the probability density function $P_\theta$. Meaning, $P_\theta$ is some differentiable function such that $P_\theta(x)\geqslant0$ and $\int_xP_\theta(x)\,\mathrm{d}x=1$. We optimize $P_\theta$ through maximum likelihood estimation.
* Learn a function that transforms an existing distribution $Z$ into $P_\theta$. Here, $g_\theta$ is some differentiable function, $Z$ is a common distribution (usually uniform or Gaussian), and $P_\theta=g_\theta(Z)$

Given function $P_\theta$, the MLE objective is

$$\displaystyle\max_{\theta\in\mathbb{R}^d}\dfrac1m\sum_{i=1}^m\log P_\theta(x^{(i)})$$

**In the limit, this is equivalent to minimizing the [KL-divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) $KL(P_r\|P_\theta)$.**

Note that if $Q(x)=0$ at an $x$ where $P(x)>0$, the KL divergence goes to $+\infty$. **This is bad for MLE if $P_\theta$ has low dimensional support, because it'll be very unlikely that all of $P_r$ lies within that support.** If even a single data point lies outside $P_\theta$'s support, the KL divergence will explode.

To deal with this, we can add random noise to $P_\theta$ when training the MLE. This ensures the distribution is defined everywhere. **But now we introduce some error, and empirically people have needed to add a lot of random noise to make models train.** That kind of sucks. **Additionally, even if we learn a good density $P_\theta$, it may be computationally expensive to sample from $P_\theta$.**

This motivates the latter approach, of learning a $g_\theta$ (a generator) to transform a known distribution $Z$. The other motivation is that it's very easy to generate samples. **Given a trained $g_\theta$, simply sample random noise $z\sim Z$, and evaluate $g_\theta(z)$.** (The downside of this approach is that we don't explicitly know what $P_\theta$, but in practice this isn't that important.)

To train $g_\theta$ (and by extension $P_\theta$), we need a measure of distance between distributions.

Different metrics (different definitions of distance) induce different sets of convergent sequences. We say distance $d$ is weaker than distance $d^{'}$ if every sequence that converges under $d^{'}$ converges under $d$.

Looping back to generative models, given a distance $d$, we can treat $d(P_r,P_\theta)$ as a loss function. **Minimizing $d(P_r,P_\theta)$ with respect to $\theta$ will bring $P_\theta$ close to $P_r$.** This is principled as long as the mapping $\theta \mapsto P_\theta$ is continuous (which will be true if $g_\theta$ is a neural net).

### Different Distances

We know we want to minimize $d$, but how do we define $d$? This section compares various distances and their properties.

On to the distances at play.

* The Total Variation (TV) distance is $$\displaystyle\delta(P_r,\,P_g)=\sup_A|P_r(A)-P_g(A)|$$.

* The Kullback-Leibler (KL) divergence is $$\displaystyle KL(P_r\|P_g)=\int_x\log\left(\dfrac{P_r(X)}{P_g(x)}\right)P_r(x)\,\mathrm{d}x$$. This isn't symmetric. The reverse KL divergence is defined as $KL(P_g\|P_r)$.

* The Jenson-Shannon (JS) divergence: Let $M$ be the mixture distribution $M=P_r/2+P_g/2$. Then $$\displaystyle JS(P_r,\,P_g)=\dfrac12KL(P_r\|P_m)+\dfrac12KL(P_g\|P_m)$$.

* Finally, the Earth Mover (EM) or Wasserstein distance: Let $\Pi(P_r,\,P_g)$ be the set of all joint distribution $\gamma$ whose marginal distributions are $P_r$ and $P_g$. Then $$\displaystyle W(P_r,\,P_g)=\inf_{\gamma\in\Pi(P_r,\,P_g)}\mathbb{E}_{(x,\,y)\sim\gamma}[\|x-y\|]$$.

  * > Probability distributions are defined by how much mass they put on each point. Imagine we started with distribution $P_r$, and wanted to move mass around to change the distribution into $P_g$. **Moving mass $m$ by distance $d$ costs $m⋅d$ effort. The earth mover distance is the minimal effort we need to spend.**
    >
    > Why does the infimum over $\Pi(P_r,\,P_g)$ give the minimal effort? You can think of each $\gamma\in\Pi$ as a transport plan. To execute the plan, for all $x,\,y$ move $\gamma(x,\,y)$ mass from $x$ to $y$.
    >
    > Every strategy for moving weight can be represented this way. But what properties does the plan need to satisfy to transform $P_r$ into $P_g$?
    >
    > * The amount of mass that leaves $x$ is $\int_y\gamma(x,y)\,\mathrm{d}y$. This must equal $P_r(x)$, the amount of mass originally at $x$.
    >
    > - The amount of mass that enters $y$ is $\int_x\gamma(x,y)\,\mathrm{d}x$. This must equal $P_g(y)$, the amount of mass that ends up at $y$.
    >
    > **This shows why the marginals of $\gamma\in\Pi$ must be $P_r$ and $P_g$.** For scoring, the effort spent is $\int_x\int_y\gamma(x,\,y)\|x-y\|\,\mathrm{d}y\mathrm{d}x=\mathbb{E}_{(x,\,y)\sim\gamma}[\|x-y\|]$. **Computing the infinum of this over all valid $\gamma$ gives the earth mover distance.**

The paper introduces a simple example to argue why we should care about the Earth-Mover distance, which shows that **there exist sequences of distributions that don’t converge under the JS, KL, reverse KL, or TV divergence, but which do converge under the EM distance.**

**This example also shows that for the JS, KL, reverse KL, and TV divergence, there are cases where the gradient is always $0$.** This is especially damning from an optimization perspective - any approach that works by taking the gradient $\nabla_\theta d(P_0,\,P_\theta)$ will fail in these cases.

### Wasserstein GAN

**Unfortunately, computing the Wasserstein distance exactly is intractable.** Let’s repeat the definition.

$$\displaystyle W(P_r,\,P_g)=\inf_{\gamma\in\Pi(P_r,\,P_g)}\mathbb{E}_{(x,\,y)\sim\gamma}[\|x-y\|]$$

The paper now shows how we can compute an approximation of this.

A result from [Kantorovich-Rubinstein duality](https://en.wikipedia.org/wiki/Wasserstein_metric#Dual_representation_of_W1) shows $W$ is equivalent to

$$\displaystyle W(P_r,\,P_\theta)=\sup_{\|f\|_L\leqslant1}\mathbb{E}_{x\sim\mathbb{P}_r}[f(x)]-\mathbb{E}_{x\sim\mathbb{P}_\theta}[f(x)]$$

where the supremum is taken over all $1$-Lipschitz functions.

If we replace the supremum over $1$-Lipschitz functions with the supremum over $K$-Lipschitz functions, then the supremum is $K \cdot W(P_r, P_\theta)$ instead. (This is true because every $K$-Lipschitz function is a $1$-Lipschitz function if you divide it by $K$, and the Wasserstein objective is linear.)

The supremum over $K$-Lipschitz functions $\{f : \|f\|_L \leqslant K\}$ is still intractable, but now it’s easier to approximate. Suppose we have a parametrized function family $\{f_w\}_{w \in \mathcal{W}}$, where $w$ are the weights and $\mathcal W$ is the set of all possible weights. Further suppose these functions are all $K$-Lipschitz for some $K$. Then we have

$$\begin{aligned}    \max_{w \in \mathcal{W}}        \mathbb{E}_{x \sim P_r}[f_w(x)] - \mathbb{E}_{x \sim P_\theta}[f_w(x)]    &\leqslant \sup_{\|f\|_L \le K}        \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_\theta}[f(x)] \\    &= K \cdot W(P_r, P_\theta) \end{aligned}$$

For optimization purposes, we don’t even need to know what $K$ is! It’s enough to know that it exists, and that it’s fixed throughout training process. **Sure, gradients of $W$ will be scaled by an unknown $K$, but they’ll also be scaled by the learning rate $\alpha$, so $K$ will get absorbed into the hyperparam tuning.**

If $\{f_w\}_{w \in \mathcal{W}}$ contains the true supremum among $K$-Lipschitz functions, this gives the distance exactly. This probably won’t be true. In that case, the approximation’s quality depends on what $K$-Lipschitz functions are missing from $\{f_w\}_{w \in \mathcal{W}}$.

Now, let’s loop all this back to generative models. We’d like to train $P_\theta = g_\theta(Z)$ to match $P_r$. Intuitively, given a fixed $g_\theta$, we can compute the optimal $f_w$ for the Wasserstein distance. We can then backprop through $W(P_r, g_\theta(Z))$ to get the gradient for $\theta$.

$$\begin{aligned}    \nabla_\theta W(P_r, P_\theta) &= \nabla_\theta (\mathbb{E}_{x \sim P_r}[f_w(x)] - \mathbb{E}_{z \sim Z}[f_w(g_\theta(z))]) \\    &= -\mathbb{E}_{z \sim Z}[\nabla_\theta f_w(g_\theta(z))] \end{aligned}$$

**The training process has now broken into three steps.**

- For a fixed $\theta$, compute an approximation of $W(P_r, P_\theta)$ by training $f_w$ to convergence.
- Once we find the optimal $f_w$, compute the $\theta$ gradient $-\mathbb{E}_{z \sim Z}[\nabla_\theta f_w(g_\theta(z))]$ by sampling several $z\sim Z$.
- Update $\theta$, and repeat the process.

There’s one final detail. This entire derivation only works when the function family $\{f_w\}_{w\in\mathcal{W}}$ is $K$-Lipschitz. To guarantee this is true, we use weight clamping. **The weights $w$ are constrained to lie within $[-c,\,c]$, by clipping $w$ after every update to $w$.**

The full algorithm is below.

![](https://www.alexirpan.com/public/wasserstein/algorithm.png)

### Empirical Results

First, the authors set up a small experiment to showcase the difference between GAN and WGAN.

There are two 1D Gaussian distributions, blue for real and green for fake. Train a GAN discriminator and WGAN critic to optimality, then plot their values over the space. The red curve is the GAN discriminator output, and the cyan curve is the WGAN critic output.

![](https://www.alexirpan.com/public/wasserstein/gauss1d.png)

Both identify which distribution is real and which is fake, but the GAN discriminator does so in a way that makes gradients vanish over most of the space. In contrast, the weight clamping in WGAN gives a reasonably nice gradient over everything.

Next, the Wasserstein loss seems to correlate well with image quality. Here, the authors plot the loss curve over time, along with the generated samples.

![](https://www.alexirpan.com/public/wasserstein/w_mlp512.png)

After reading through the paper, this isn’t too surprising. Since we’re training the critic $f_w$ to convergence, these critic’s value should be good approximations of $K \cdot W(P_r, P_\theta)$, where $K$ is whatever the Lipschitz constant is. As argued before, a low $W(P_r, P_\theta)$ means $P_r$ and $P_\theta$ are “close” to one another. It would be more surprising if the critic value *didn’t* correspond to visual similarity.

The image results also look quite good. Compared to the DCGAN baseline on the bedroom dataset, it performs about as well.

![WGAN with DCGAN architecture](https://www.alexirpan.com/public/wasserstein/wgan_bn.png)

![DCGAN with DCGAN architecture](https://www.alexirpan.com/public/wasserstein/dcgan_bn.png)

If we remove batch norm from the generator, WGAN still generates okay samples, but DCGAN fails completely.

![WGAN with DCGAN architecture, no batch norm](https://www.alexirpan.com/public/wasserstein/wgan_nobn.png)

![DCGAN with DCGAN architecture, no batch norm](https://www.alexirpan.com/public/wasserstein/dcgan_nobn.png)

Finally, we make the generator a feedforward net instead of a convolutional one. This keeps the number of parameters the same, while removing the inductive bias from convolutional models. The WGAN samples are more detailed, and don’t mode collapse as much as standard GAN. In fact, they report never running into mode collapse at all for WGANs!

![WGAN with MLP architecture](https://www.alexirpan.com/public/wasserstein/wgan_mlp.png)

![DCGAN with MLP architecture](https://www.alexirpan.com/public/wasserstein/gan_mlp.png)