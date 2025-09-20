# Improving Single Noise Level Diffusion Samplers with Restricted Gaussian Oracles

ICLR 2025 DeLTa Workshop Poster

----------------------

## Abstract

Diffusion models and diffusion Monte-Carlo schemes that sample from unnormalized log-densities, both rely on denoisers (
or score estimates) at different noise scales. This complicates the sampling process as denoising schedules require
careful tuning and nested inner-MCMC loops. In this work, we propose a single noise level sampling procedure that only
requires a single low-noise denoiser. Our framework results from improvements we bring to the multimeasurement Walk-Jump
sampler of Saremi et al. 2021 by mixing in ideas from the proximal sampler of Shen et al. 2020. Our analysis shows that
annealing (or multiple noise scales) is unnecessary if one is willing to pay an increased memory cost. We demonstrate
this by proposing an *entirely log-concave* sampling framework.

## Instructions

### Obtaining the code

Clone this repository by running:

`git clone https://github.com/Bani57/multi-prox-diffusion-iclr-delta-2025.git`

### Dependencies

This repository extends the implementation of "DiGress: Discrete Denoising diffusion models for graph generation",
originally available at https://github.com/cvignac/DiGress.

To install the Python programming language and the dependent Python packages, you need to execute the following steps:

- Download anaconda/miniconda if needed

- Create a rdkit environment that directly contains rdkit:

  ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```

- `conda activate digress`

- Check that this line does not return an error:

  ``` python3 -c 'from rdkit import Chem' ```

- Install graph-tool (https://graph-tool.skewed.de/):

  ```conda install -c conda-forge graph-tool=2.45```

- Check that this line does not return an error:

  ```python3 -c 'import graph_tool as gt' ```

- Install the nvcc drivers for your cuda version. For example:

  ```conda install -c "nvidia/label/cuda-11.8.0" cuda```

- Install a corresponding version of pytorch, for example:

  ```pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118```

- Install other packages using the requirement file:

  ```pip install -r requirements.txt```

- Run:

  ```pip install -e .```

- Navigate to the `src/analysis/orca` directory and compile `orca.cpp`:

  ```g++ -O2 -std=c++11 -o orca orca.cpp```

### Reproducing results

To generate molecules with our Multi-prox sampler and obtain example chains like the one displayed in our work, run the
following command from inside the `src` directory:

```CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py dataset="qm9" general.name="c_gibbs_startNone_N10_M100_t0.5_0.1" model="continuous" general.test_only="<REPO_ABSPATH>/checkpoints/qm9_c.ckpt" +model.gibbs=True +model.gibbs_N=10 +model.gibbs_M=100 +model.gibbs_fixed_t_2=0.5 +model.gibbs_fixed_t_1=0.1 +model.gibbs_chain_freq=0.1 ++general.final_model_samples_to_generate=1000```

where you would replace `<GPU_ID>` with the integer ID of one of your available GPU devices and replace `<REPO_ABSPATH>`
with the absolute path of the main repo directory (where you cloned it).

The batch of full molecule chains will be saved to
`<REPO_ABSPATH>/chains/c_gibbs_startNone_N10_M100_t0.5_0.1_resume/epoch0/chains/`, while the batch of just the final
molecules will be saved to `<REPO_ABSPATH>/graphs/c_gibbs_startNone_N10_M100_t0.5_0.1_resume/epoch0_b0/`.

To reproduce our results with the denoising diffusion GAN for images, run these commands from inside the `src`
directory:

- CIFAR 10:
  ```CUDA_VISIBLE_DEVICES=<GPU_ID> python main_cv.py --dataset cifar10 --exp ddgan_cifar10_exp1 --num_channels 3 --num_channels_dae 128 --num_timesteps 4 --num_res_blocks 2 --nz 100 --z_emb_dim 256 --n_mlp 4 --ch_mult 1 2 2 2 --epoch_id 1200 --n 1 --m 10 --fixed_t 0.25 --sample_freq 1 ```

- CelebA-256:
  ```CUDA_VISIBLE_DEVICES=<GPU_ID> python main_cv.py --dataset celeba_256 --image_size 256 --batch_size 50 --exp ddgan_celebahq_exp1 --num_channels 3 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 2 --num_res_blocks 2 --epoch_id 550 --n 1 --m 10 --fixed_t 0.5 --sample_freq 1```

The generated image chains will be saved to `<REPO_ABSPATH>/generated_chains` and the individual images to
`<REPO_ABSPATH>/generated_samples`.

To reproduce our Mixture of Gaussians RGO example, run the `MoG40.ipynb` notebook.

----------

## Citation

If you use this code for your projects, please cite:
```
@inproceedings{
dadi2025improving,
title={Improving Single Noise Level Denoising Samplers with Restricted Gaussian Oracles},
author={Leello Tadesse Dadi and Andrej Janchevski and Volkan Cevher},
booktitle={ICLR 2025 Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy},
year={2025},
url={https://openreview.net/forum?id=xkiI5tou6J}
}
```

----------

## Authors

- Leello Tadesse Dadi, EPFL STI IEM LIONS, Lausanne, Switzerland, leello.dadi@epfl.ch
- Andrej Janchevski, EPFL STI IEM LIONS, Lausanne, Switzerland, andrej.janchevski@epfl.ch
- Volkan Cevher, EPFL STI IEM LIONS, Lausanne, Switzerland, volkan.cevher@epfl.ch
