## TinyGAN
BigGAN; Knowledge Distillation; Black-Box; Fast Training; 16x compression

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch 1.2.0](https://img.shields.io/badge/pytorch-1.2.0-green.svg?style=plastic)

This repository contains the official PyTorch implementation of the following paper:

> **TinyGAN: Distilling BigGAN for Conditional Image Generation (ACCV 2020)**<br>
> Ting-Yun Chang and Chi-Jen Lu<br>

>
> **Abstract:** *Generative Adversarial Networks (GANs) have become a powerful approach for generative image modeling. However, GANs are notorious for their training instability, especially on large-scale, complex datasets. While the recent work of BigGAN has significantly improved the quality of image generation on ImageNet, it requires a huge model, making it hard to deploy on resource-constrained devices. To reduce the model size, we propose a black-box knowledge distillation framework for compressing GANs, which has a stable and efficient training process. Given BigGAN as the teacher network, we manage to train a much smaller student network to mimic its functionality, which achieves competitive performance on Inception and FID scores but with the generator having 16 times fewer parameters.*

### Training
```bash
$ bash train.sh
```

### Evaluation
```bash
$ bash eval.sh
```

---
![Fig](https://terarachang.github.io/files/TinyGAN_flow.png)
---
![Fig](https://terarachang.github.io/files/TinyGAN_demo.png)
