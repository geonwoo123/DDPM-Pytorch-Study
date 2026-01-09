# DDPM: Code Analysis & Implementation Study

> **Paper:** [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)  
> **Original Code Source:** [The Annotated Diffusion Model (Hugging Face)](https://huggingface.co/blog/annotated-diffusion)

## 1. Project Overview
This repository is dedicated to a deep dive into the architecture and mathematics of **DDPM**.
Instead of simply running the model, I focused on **analyzing the correspondence between the paper's mathematical formulas and the actual PyTorch implementation**. 

I have added detailed **line-by-line comments** to the source code and documented my study process to demonstrate a thorough understanding of the logic.

*   **Objective:** Mathematical verification & Code logic analysis.
*   **Method:** Annotated the original source code and created detailed study notes.

---

## 2. 📝 Study Notes & Annotations
To fully understand the mathematical derivation (especially the Reverse Process), I studied using existing lecture materials and added my own analysis.

*   **Reference Material:** [Name of the Lecture/Slide Author] (e.g., *CVPR 2022 Tutorial on Diffusion Models*)
*   **My Contribution:** Added handwritten notes explaining the derivation of the ELBO and variance scheduling.

> **👇 Click to view my full study log:**
> 
> 📎 **[View Annotated Study Notes (PDF)](./docs/DDPM_Study_Notes.pdf)**

**[Preview of My Analysis]**
*(I focused on why the model predicts noise $\epsilon$ instead of $x_0$ directly)*

<img src="./docs/note_preview.png" width="700" alt="Study Note Preview">
<!-- ⚠️ Replace 'note_preview.png' with your actual screenshot filename -->

---

## 3. Core Logic Analysis
*The code below contains my personal annotations explaining the logic.*

### 3.1 Forward Process (Diffusion)
The process of gradually adding Gaussian noise to the image.

$$ q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}) $$

```python
# [PASTE YOUR CODE HERE]
# Paste your 'q_sample' code with your comments.
# Example:
# def q_sample(x_start, t, noise=None):
#     ...
```

### 3.2 Reverse Process (Denoising)
The neural network approximates the reverse process to recover the image. 

$$ p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)) $$

```python
# [PASTE YOUR CODE HERE]
# Paste your 'p_sample' code with your comments.
```

### 3.3 Loss Function
Optimizing the variational lower bound simplifies to minimizing the MSE between actual noise and predicted noise.

$$ L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t) \|^2 \right] $$

```python
# [PASTE YOUR CODE HERE]
# Paste your loss calculation code (p_losses) here.
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
