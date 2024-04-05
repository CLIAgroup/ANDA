# Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning (Accepted by CVPR2024)

This is the official PyTorch code for ANDA and MultiANDA in our paper [Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning](https://arxiv.org/pdf/2209.11964.pdf).

# Abstract
[![arXiv](https://img.shields.io/badge/arXiv-2209.11964-b31b1b.svg)](https://arxiv.org/abs/2209.11964)  
Strong adversarial examples are crucial for evaluating and enhancing the robustness of deep neural networks. However, the performance of popular attacks is usually sensitive, for instance, to minor image transformations, stemming from limited information -- typically only one input example, a handful of white-box source models, and undefined defense strategies. Hence, the crafted adversarial examples are prone to overfit the source model, which hampers their transferability to unknown architectures. In this paper, we propose an approach named Multiple Asymptotically Normal Distribution Attacks (MultiANDA) which explicitly characterize adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then employ the deep ensemble strategy as an effective proxy for Bayesian marginalization in this process, aiming to estimate a mixture of Gaussians that facilitates a more thorough exploration of the potential optimization space. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, MultiANDA allows drawing an unlimited number of adversarial perturbations for each input and reliably maintains the transferability. Our proposed method outperforms ten state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defense models.

# Requirements
- Python == 3.10.10
- torch == 2.0.1
- torchvision == 0.15.2
- numpy == 1.25.2
- scipy == 1.11.1

# Datasets
Please download `datasets.zip` from [ImageNet Subset](https://drive.google.com/file/d/1jXwCXyUc0R4j_i-ZLhrzel48XqzpDYry/view?usp=drive_link), and unzip it as `./datasets`

# Run the code
Please refer to the `scripts/` dir for how to run the code.
- ANDA and ANDA-CT in `scripts/anda.sh`
- MultiANDA and MultiANDA-CT in `scripts/multianda.sh` (implemented by torch.distributed)

# Notes
1. Only support pytorch official model ckpts in this repo (for a simple demonstration), if you need to reproduce the results of our paper, please refer to the detailed model information and source listed below: 

    - vgg19 ([pytorch offical](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html))
    - tf_inc_res_v2, tf_resnet_v2_50, tf_inception_v3, tf_ens3_adv_inc_v3, tf_ens_adv_inc_res_v2 ([tf2torch](https://github.com/ylhz/tf_to_pytorch_model))
    - [HGD](https://github.com/lfz/Guided-Denoise), [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding repo.
    - [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc_v3_ens

2. Output option (b) in Algorithm-(1/A.1) can be enabled by setting `sample=True` in `attack` function in `anda.py/anda_ct.py/multianda.py/multianda_ct.py`.

3. The CT version of ANDA/MultiANDA may not be CUDA memory friendly, one can enable minibatch forward to save CUDA memory by setting `minibatch=True` of `attack` function in `attack/anda_ct.py` and `attack/multianda_ct.py`, default `minibatch=False`.


# Acknowledgments
We are very grateful for the contribution of following efforts to the community:
- https://github.com/ylhz/tf_to_pytorch_model
- https://github.com/JHL-HUST/VT

# Citation
If you find this work is useful, please consider citing:
```
@article{fang2022approximate,
  title={Approximate better, Attack stronger: Adversarial Example Generation via Asymptotically Gaussian Mixture Distribution},
  author={Fang, Zhengwei and Wang, Rui and Huang, Tao and Jing, Liping},
  journal={arXiv preprint arXiv:2209.11964},
  year={2022}
}
```

# Contact
Please feel free to email `jankinfmail@gmail.com`.