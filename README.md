# DDPM: Code Analysis & Implementation Study

> **Paper:** [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)  
> **Original Code Source:** [The Annotated Diffusion Model (Hugging Face)](https://huggingface.co/blog/annotated-diffusion)

## 1. Project Overview
This repository is dedicated to a deep dive into the architecture and mathematics of DDPM. Instead of simply running the model, I focused on analyzing the correspondence between the paper's mathematical formulas and the actual PyTorch implementation.

I have added detailed line-by-line comments to the source code to demonstrate a thorough understanding of the logic. **Furthermore, based on this analysis, I utilized the code as a framework to implement a custom generation task. I sourced the "Pokemon Image Dataset" from Kaggle and modified the model to successfully generate new Pokemon images.**

*   **Objective:** Mathematical verification, Code logic analysis, and **Practical Application (Custom Dataset Training).**
*   **Method:**
    1.  Annotated the original source code with mathematical explanations.
    2.  **Refactored the code to train on the Kaggle Pokemon dataset and generated RGB images.**

---

## 2. 📝 Study Notes & Annotations
To fully understand the mathematical derivation, I utilized existing lecture materials and added my own analysis directly onto the slides.

*   **Reference Material:** [Name of the Lecture/Slide Author] (e.g., CVPR 2022 Tutorial on Diffusion Models)
*   **My Contribution:** I annotated the original slides with **handwritten notes**, specifically focusing on a detailed analysis of the mathematical significance behind the **Forward and Reverse process equations**.

### 👁️ Preview
<!-- 아래에 캡처한 이미지 파일 이름(note_preview.png)이 정확해야 합니다 -->
<img src="./note_preview.png" alt="Study Note Preview" width="700" style="border: 1px solid #ddd; border-radius: 5px;">

<br>

👇 **Want to see more details?**

📎 [**Click here to view the FULL Study Log (PDF)**](./my_study_notes.pdf)

---

## 3. Core Logic Analysis
*The code below contains my personal annotations explaining the logic.*

### 3.1 Forward Process (Diffusion)
The process of gradually adding Gaussian noise to the image.

$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}) $$

```python
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
```

### 3.2 Reverse Process (Denoising)
The neural network approximates the reverse process to recover the image. 

$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

```python
@torch.no_grad()                   
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:                                            
        return model_mean                                       
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape) 
        noise = torch.randn_like(x)                            
        # 알고리즘 2, 4번째 줄: 평균 + σ_t * ε  (ε ~ N(0, I))
        return model_mean + torch.sqrt(posterior_variance_t) * noise
```

### 3.3 Loss Function
Optimizing the variational lower bound simplifies to minimizing the MSE between actual noise and predicted noise.

$$ L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t) \|^2 \right] $$

```python
def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    predicted_noise = denoise_model(x_noisy, t)

    loss = F.mse_loss(noise, predicted_noise)

    return loss
```

---

## 4. Key Learnings & Troubleshooting

### 🔍 Tensor Dimension Mismatch
*   **Issue:** Encountered `RuntimeError` during the broadcasting step when adding noise.
*   **Analysis:** The 1D time parameters needed to be reshaped to match the 4D image tensor $(B, C, H, W)$.
*   **Solution:** Utilized `view()` and `reshape()` to ensure correct broadcasting.

### 💡 Understanding the Objective
*   **Insight:** Initially, I assumed the model reconstructs the image $x_0$ directly.
*   **Correction:** Through code analysis, I realized the model predicts the **noise component $\epsilon$**. This makes the training objective mathematically equivalent to score matching and provides more stable gradients.

---

## 5. Generation Results

I verified the model's performance by testing it on two different datasets.
First, I reproduced the results using the standard Fashion MNIST dataset to verify the baseline logic. Then, **I modified the code to support RGB channels** and trained it on a custom Pokemon dataset.

| **Baseline: Fashion MNIST** | **Application: Pokemon (Custom)** |
| :---: | :---: |
| <img src="./results/fashion_mnist_sample.png" width="300" alt="Fashion MNIST"> | <img src="./ddpm_epoch100_poketmon.png" width="300" alt="Pokemon Generated"> |
| *Initial verification (Grayscale, 28x28)* | *Code modified for RGB (Epoch 100)* |

### 🛠️ Code Modifications & Analysis
To transition from Fashion MNIST to the Pokemon dataset, I made the following adjustments to the U-Net architecture:

1.  **Channel Adaptation:**
    *   Changed input/output channels from `1` (Grayscale) to `3` (RGB).
    *   Modified the `Unet` class initialization: `channels=3`.
2.  **Data Preprocessing:**
    *   Updated `torchvision.transforms` to normalize RGB values to $[-1, 1]$.
3.  **Result Analysis (Pokemon):**
    *   The generated image (Right) shows features resembling a mix of *Cubone (탕구리)* and *Kangaskhan*, indicating the model successfully learned the color distribution and morphological features of the dataset.|
