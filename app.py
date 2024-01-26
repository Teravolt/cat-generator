# """
# Gradio app for generating cat pictures
# """

from argparse import Namespace

import os

import numpy as np
import gradio as gr

from diffusers import UNet2DModel
from diffusers import DDIMScheduler

import torchvision
from PIL import Image

import wandb

import torch
from torch import nn

SCALE_FACTOR = 2
UPSAMPLER = nn.Upsample(
    scale_factor=SCALE_FACTOR,
    mode='nearest')

# MODEL_ARTIFACT = './artifacts/cat-dataset-10K-baseline-model:v0'
MODEL_ARTIFACT = './artifacts/cat-dataset-model-v2:v6'

if not os.path.exists(MODEL_ARTIFACT):
    run = wandb.init()
    # artifact = run.use_artifact('pkthunder/Cat-Generator/cat-dataset-10K-baseline-model:v0', type='model')
    # artifact = run.use_artifact('pkthunder/Cat-Generator/cat-dataset-baseline-model:v0', type='model')
    # artifact = run.use_artifact('pkthunder/Cat-Generator/cat-dataset-model:v1', type='model')
    artifact = run.use_artifact('pkthunder/Cat-Generator/cat-dataset-model-v2:v6', type='model')
    artifact_dir = artifact.download()
    run.finish()

SEED = 1

CONFIG = Namespace(
    run_name='cat-diffusion-model-scratch',
    model_name='cat-dataset-model-v2',
    image_size=128,
    num_samples_to_generate=8,
    horizontal_flip_prob=0.5,
    gaussian_blur_kernel_size=3,
    per_device_train_batch_size=8,
    num_train_epochs=15,
    learning_rate=4e-4,
    seed=SEED,
    num_train_timesteps=1000,
    beta_schedule='squaredcos_cap_v2',
    lr_exp_schedule_gamma=0.85,
    lr_warmup_steps=500,
    train_limit=-1
    )

MODEL = UNet2DModel(
    sample_size=CONFIG.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)

MODEL.load_state_dict(torch.load(f"{MODEL_ARTIFACT}/model.pt", map_location=torch.device('cpu')))

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'mps' \
        if torch.backends.mps.is_available() else 'cpu')
MODEL.to(DEVICE)

# NOISE_SCHEDULER = DDPMScheduler(
#     num_train_timesteps=CONFIG.num_train_timesteps,
#     beta_schedule=CONFIG.beta_schedule)


NOISE_SCHEDULER = DDIMScheduler(
    num_train_timesteps=CONFIG.num_train_timesteps,
    beta_schedule=CONFIG.beta_schedule)
NOISE_SCHEDULER.set_timesteps(num_inference_steps=500)

def generate_cat(seed: int):
    """
    Generate a cat given a random seed
    """

    rng = torch.Generator()
    rng = rng.manual_seed(seed)

    sample = torch.randn(4, 3, CONFIG.image_size, CONFIG.image_size, generator=rng)
    sample = sample.to(DEVICE)

    for _, t in enumerate(NOISE_SCHEDULER.timesteps):

        # Get model pred
        with torch.no_grad():
            residual = MODEL(sample, t).sample

        # Update sample with step
        sample = NOISE_SCHEDULER.step(residual, t, sample).prev_sample

    sample = sample * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    # sample = UPSAMPLER(sample)

    grid = torchvision.utils.make_grid(sample)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im

demo = gr.Interface(
    generate_cat,
    inputs=gr.Slider(0, 1000, label='Seed'),
    outputs="image")

demo.launch()
