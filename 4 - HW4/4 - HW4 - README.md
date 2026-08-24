# Chapter 4: Generative Models — VAEs, GANs, and Diffusion Models  
In this chapter, we explore the world of **generative models**, including **Variational Autoencoders (VAEs)**, **Generative Adversarial Networks (GANs)**, and **Diffusion Models**.  
This chapter provided a strong theoretical foundation while also offering far more practical and coding potential than I initially expected. Each of these model families introduces a different approach to learning data distributions and generating new samples.  
Because generative models can be computationally expensive to train, I used cloud-based GPU resources during the practical parts of this chapter, including:  
- **NVIDIA Tesla T4 GPUs on Kaggle**
- **NVIDIA A100 GPU through Google Colab Pro**  
These resources made it possible to train and experiment with the models more efficiently.  
Throughout this chapter, I also saved the trained models best-performing weights, and the generated results. The following sections include the implementations, experiments, and outputs for each generative modeling approach.  

> **Note**
>
> This chapter is divided into two sections:
>
> - Theoretical
> - Practical

---
# ✒️ Theoretical Part

The theoretical section of this chapter covers the core concepts behind modern generative modeling, including:  
- Step-by-step design and formulation of **Variational Autoencoders (VAEs)**
- **Hierarchical VAEs**
- **Generative Adversarial Networks (GANs)**
- **Score-Based Learning** in Diffusion Models
- The **Markov Process** underlying Diffusion Models

---

# 👨‍💻 Practical Part  
The practical part consists of two parts, which are presented below.  

## 1. Variational Autoencoders (VAEs)

In this part of the project, we explore several stages and extensions of the **Variational Autoencoder (VAE) family**. The models are trained and evaluated mainly on the `Fashion-MNIST` dataset, allowing us to study how latent representations evolve from deterministic Autoencoders to probabilistic VAEs and how these learned representations can later be used for generation, clustering, classification, and transfer learning.

### 1.1. Autoencoder (AE)

We first implement a standard **Autoencoder (AE)** as the deterministic baseline of the VAE family.

The Autoencoder learns to compress an input image into a lower-dimensional latent representation and reconstruct the original image from this representation:

$$
x \rightarrow \text{Encoder} \rightarrow z \rightarrow \text{Decoder} \rightarrow \hat{x}
$$

Unlike a VAE, the latent representation $z$ is deterministic and is not explicitly constrained to follow a predefined probability distribution.

---

### 1.2. Variational Autoencoder (VAE)

We then extend the standard Autoencoder into a **Variational Autoencoder** by replacing the deterministic latent representation with a probabilistic latent space.

Instead of directly predicting a latent vector $z$, the encoder predicts the parameters of a Gaussian distribution:

$$
q_\phi(z|x) = \mathcal{N}\left(\mu,\sigma^2\right)
$$

The **reparameterization trick** is then used to sample from this distribution while preserving gradient-based optimization:

$$
z = \mu + \sigma \odot \epsilon,
\quad \text{where} \quad
\epsilon \sim \mathcal{N}(0,I)
$$

This probabilistic latent space makes random sampling and image generation possible.

After training, latent vectors can be sampled from the prior distribution:

$$
z \sim \mathcal{N}(0,I)
$$

and decoded into new Fashion-MNIST-like images.

<p align="center">
  <img src="Practical\Results\VAE\10 generated samples.png" alt="10 generated samples by VAE" width="600">
  <br>
  <em>Figure 1: Ten generated samples produced by the VAE.</em>
</p>

---

### 1.3. KL Divergence in the VAE

A major difference between a standard Autoencoder and a VAE is the addition of the **Kullback-Leibler (KL) Divergence** term.

The KL divergence encourages the learned latent distribution to remain close to a standard normal prior:

$$
p(z)=\mathcal{N}(0,I)
$$

The complete VAE objective consists of two components:

$$
\mathcal{L}_{VAE} = \mathcal{L}_{reconstruction} + D_{KL}\left(q_\phi(z|x)\parallel p(z)\right)
$$

The reconstruction term encourages accurate reconstruction of the input, while the KL term regularizes the latent space and creates a smoother and more structured distribution that supports meaningful random sampling.

---

### 1.4. Traversing Latent Dimensions in VAE

In this experiment, we investigate how individual dimensions of the **latent space** influence the generated images.

Different values are systematically sampled along selected latent dimensions while the remaining dimensions are kept fixed. All combinations are then passed through the decoder.

This allows us to visually inspect how movement through the latent space changes different characteristics of the generated data.

Conceptually:

$$
z=[z_1,z_2,\ldots,z_d]
$$

By changing selected values such as $z_1$ and $z_2$, we can observe how the decoder responds to different regions of the learned latent space.

<p align="center">
  <img src="Practical\Results\VAE\Axis in VAE.png" alt="Traversing latent dimensions in VAE" width="600">
  <br>
  <em>Figure 2: Traversing different regions of the VAE latent space.</em>
</p>

---

### 1.5. Clustering the Latent Space vs. the Original Data Classes

In this section, we investigate whether the latent representations learned by the VAE contain meaningful class-related structure.

For every input image, the encoder produces a latent representation:

$$
x\rightarrow\text{Encoder}\rightarrow\mu
$$

These latent vectors are then clustered using **K-Means Clustering** without using the original class labels.

The discovered clusters are compared with the true `Fashion-MNIST` classes to examine whether samples belonging to similar semantic categories are also located close to each other in the learned latent space.

Because the latent representation is high-dimensional, dimensionality reduction is used for visualization.

<p align="center">
  <img src="Practical\Results\VAE\Clustering.png" alt="Clustering the VAE latent space" width="600">
  <br>
  <em>Figure 3: Comparison between true Fashion-MNIST classes and clusters discovered in the VAE latent space.</em>
</p>

The results show a noticeable similarity between several discovered clusters and the actual data classes, suggesting that the VAE has learned meaningful structural representations of the dataset.

---

### 1.6. Adding a Classifier to the VAE

In this experiment, we reuse the latent representation learned by the VAE for a **classification task**.

A classification head is added on top of the encoder:

$$
x
\rightarrow
\text{VAE Encoder}
\rightarrow
\mu
\rightarrow
\text{Classifier}
\rightarrow
\hat{y}
$$

The classifier is then fine-tuned using the original Fashion-MNIST class labels.

This experiment demonstrates that the latent features learned by the generative model can also be useful for discriminative downstream tasks.

The final model achieved a test accuracy of:

$$
\mathbf{85.61\%}
$$

on the Fashion-MNIST test set.

---

### 1.7. Adversarial Examples in Variational Autoencoders

In this experiment, we investigate the sensitivity of the VAE to **adversarial perturbations**.

A small, intentionally designed perturbation is added to an input image using the **Fast Gradient Sign Method (FGSM)**:

$$
x_{\text{adv}} = x + \epsilon \cdot \mathrm{sign}\left(\nabla_x L\right)
$$

Although the adversarial image may still appear visually similar to the original input, the carefully chosen perturbation can significantly affect the reconstruction produced by the VAE.

<p align="center">
  <img src="Practical\Results\VAE\Adversarial Examples in VAE.png" alt="Adversarial example in VAE" width="600">
  <br>
  <em>Figure 4: Original and adversarial inputs together with their corresponding VAE reconstructions.</em>
</p>

The resulting reconstructions demonstrate that even relatively small perturbations can noticeably affect the output of the model.

---

### 1.8. Fine-Tuning VAE for MNIST Digit Classification

Finally, we explore **cross-domain representation learning** by transferring a VAE originally trained on `Fashion-MNIST` to the `MNIST` digit classification task.

The Fashion-MNIST-trained encoder is reused as a feature extractor, and a classification head is trained on MNIST digit labels.

Although the original VAE never encountered handwritten digits during its initial training, its encoder learned reusable low-level and mid-level visual features such as:

- Edges
- Curves
- Strokes
- Shapes
- Textures

The experiment tests whether these learned representations can be transferred to a visually different dataset.

The resulting model achieved a test accuracy of:

$$
\mathbf{95.41\%}
$$

on the `MNIST` test set.

This result demonstrates that the representations learned by the VAE on Fashion-MNIST can be effectively reused for a different visual domain, highlighting the transferability of learned latent features.


## 2. Denoising Diffusion Probabilistic Models (DDPMs)

In this section, we explore **Denoising Diffusion Probabilistic Models (DDPMs)** based on the ideas introduced in the original diffusion and classifier-free guidance papers:

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/pdf/2006.11239.pdf)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

The main objective of this part was to implement and train both **unconditional** and **conditional diffusion models** on the `MNIST` dataset and investigate the practical complexity behind diffusion-based generative modeling.

### 2.1. Forward Diffusion Process

The forward diffusion process gradually adds Gaussian noise to the original image over a sequence of timesteps.

Instead of applying the entire noising process step by step during training, a noisy image at an arbitrary timestep can be obtained directly using:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

where:

- $x_0$ is the original image.
- $x_t$ is the noisy image at timestep $t$.
- $\epsilon \sim \mathcal{N}(0,I)$ is Gaussian noise.
- $\bar{\alpha}_t$ controls the amount of original signal remaining at timestep $t$.

The network is then trained to predict the noise that was added to the image.

---

### 2.2. U-Net Noise Prediction Network

A **U-Net-based architecture** is used as the main noise prediction network.

The U-Net contains:

- Downsampling blocks
- Upsampling blocks
- Residual connections
- Skip connections between encoder and decoder stages
- Self-attention blocks
- Timestep conditioning

The model receives a noisy image $x_t$ together with its timestep $t$ and predicts the Gaussian noise:

$$
\epsilon_\theta(x_t,t)
$$

The training objective is based on the difference between the actual noise and the predicted noise:

$$
\mathcal{L} = \mathbb{E}\left[\left\|\epsilon-\epsilon_\theta(x_t,t)\right\|^2\right]
$$

Residual connections are used throughout the network to improve information flow and stabilize training, while attention blocks allow the network to model relationships between different spatial regions of the image.

---

### 2.3. Timestep Embedding

One of the most important and initially challenging concepts in this section was **timestep embedding**.

The amount of noise contained in an image depends heavily on the current diffusion timestep. Therefore, the U-Net must know **where the current sample is located in the diffusion process**.

A sinusoidal embedding is used to represent each timestep at multiple frequencies:

$$
\omega_i = \mathrm{max\_period}^{-\frac{2i}{d}}
$$

The resulting timestep representation is passed through the network and injected into the residual blocks.

Conceptually, the model performs:

$$
(x_t,t)
\rightarrow
\text{U-Net}
\rightarrow
\hat{\epsilon}
$$

This practical implementation made the importance of timestep conditioning much clearer than the theoretical formulation alone. The network does not simply receive a noisy image; it must also understand **how noisy that image is expected to be at the corresponding timestep**.

---

### 2.4. Unconditional DDPM

The first implementation is an **unconditional diffusion model**.

The model learns the general distribution of MNIST images without receiving information about the desired digit class.

During generation, sampling begins from pure Gaussian noise:

$$
x_T \sim \mathcal{N}(0,I)
$$

The model then repeatedly predicts and removes noise:

$$
x_T
\rightarrow
x_{T-1}
\rightarrow
\cdots
\rightarrow
x_1
\rightarrow
x_0
$$

until a recognizable digit image is generated.

---

### 2.5. Conditional DDPM and Classifier-Free Guidance

The diffusion model is then extended into a **conditional DDPM**, where the digit class is provided as additional information.

This allows the model to generate a specific class, such as a `3`, `7`, or `9`, rather than generating an arbitrary MNIST digit.

To implement **Classifier-Free Guidance (CFG)**, the model is trained using both:

- Conditional examples
- Unconditional examples

During training, the class condition is randomly removed for a fraction of samples. This allows the same network to learn both conditional and unconditional noise predictions.

During generation, the two predictions are combined using:

$$
\epsilon_{\text{guided}} =
\epsilon_{\text{uncond}} +
w\left(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}}\right)
$$

where $w$ represents the guidance strength.

This enables stronger control over which digit class should be generated without requiring a separate classifier. and here in figure 5 you can see samples of the generated conditional DDPM.

<p align="center">
  <img src="Practical\Results\DDPM\conditional_ddpm_samples_2.png" alt="Conditional DDPM" width="600">
  <br>
  <em>Figure 5: Ten generated samples produced by the Condtional DDPM, each row for each class.</em>
</p>

---

### 2.6. Model Evaluation Using FID

After training and evaluating the conditional and unconditional DDPMs, the quality of the generated samples was evaluated using the **Fréchet Inception Distance (FID)**.

FID compares the feature distributions of real and generated images. In general, a lower FID score indicates that the generated distribution is closer to the real data distribution.

The resulting FID score obtained in this experiment was:

$$
\boxed{\text{FID} = 164.3046}
$$

This evaluation provides a quantitative measurement of the difference between generated MNIST samples and real MNIST images.

It is worth noting that standard FID uses features from an Inception network originally trained on natural RGB images, while MNIST consists of small grayscale digit images. Therefore, the score is useful as an experimental comparison metric in this project, but it should not be interpreted in exactly the same way as FID values reported for large natural-image datasets.

---

Overall, this section demonstrated that diffusion models are considerably more computationally and architecturally involved than simpler generative approaches.

The complete process combines:

$$
\begin{aligned}
\epsilon_{\text{guided}}
&= \epsilon_{\text{uncond}} \\
&\quad + w\left(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}}\right)
\end{aligned}
$$

Implementing the model from the architectural components upward provided a much clearer understanding of how the theoretical diffusion equations translate into an actual generative system.

---

---

# 📂 Repository Contents

## `DDPM`:
### 🤖 Trained Models
- `conditional_unet.pth`: the best weight of rhe DDPM for conditional model
- `unconditional_unet.pth`: the best weight of rhe DDPM for Unconditional model

---

# 🛠 Requirements

Install the required dependencies before running the notebooks:

```bash
pip install torchmetrics
```







