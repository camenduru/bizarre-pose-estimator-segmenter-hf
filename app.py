#!/usr/bin/env python

from __future__ import annotations

import functools
import os
import pathlib
import sys
import tarfile
from typing import Callable

import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

sys.path.insert(0, 'bizarre-pose-estimator')

from _util.twodee_v0 import I as ImageWrapper

DESCRIPTION = '# [ShuhongChen/bizarre-pose-estimator (segmenter)](https://github.com/ShuhongChen/bizarre-pose-estimator)'


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset')
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model(
        device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    path = huggingface_hub.hf_hub_download(
        'public-data/bizarre-pose-estimator-models', 'segmenter.pth')
    ckpt = torch.load(path)

    model = torchvision.models.segmentation.deeplabv3_resnet101()
    model.classifier = nn.Sequential(
        torchvision.models.segmentation.deeplabv3.ASPP(2048, [12, 24, 36]),
        nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
    )
    final_head = nn.Sequential(
        nn.Conv2d(16 + 3, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(8),
        nn.LeakyReLU(),
        nn.Conv2d(8, 2, kernel_size=1, stride=1),
    )
    model.load_state_dict(ckpt['model'])
    final_head.load_state_dict(ckpt['final_head'])
    model.to(device)
    model.eval()
    final_head.to(device)
    final_head.eval()
    return model, final_head


@torch.inference_mode()
def predict(image: PIL.Image.Image, score_threshold: float,
            transform: Callable, device: torch.device, model: torch.nn.Module,
            final_head: torch.nn.Module) -> np.ndarray:
    data = ImageWrapper(image).resize_min(256).convert('RGBA').alpha_bg(
        1).convert('RGB').pil()
    data = torchvision.transforms.functional.to_tensor(data)
    data = transform(data)
    data = data.to(device).unsqueeze(0)

    out = model(data)['out']
    out_fin = final_head(torch.cat([
        out,
        data,
    ], dim=1))
    probs = torch.softmax(out_fin, dim=1)[0]
    probs = probs[1]  # foreground
    probs = PIL.Image.fromarray(probs.cpu().numpy()).resize(image.size)

    mask = np.asarray(probs).copy()
    mask[mask < score_threshold] = 0
    mask[mask > 0] = 1
    mask = mask.astype(bool)

    res = np.asarray(image).copy()
    res[~mask] = 255
    return res


image_paths = load_sample_image_paths()
examples = [[path.as_posix(), 0.5] for path in image_paths]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model, final_head = load_model(device)
transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

fn = functools.partial(predict,
                       transform=transform,
                       device=device,
                       model=model,
                       final_head=final_head)

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Input', type='pil')
            threshold = gr.Slider(label='Score Threshold',
                                  minimum=0,
                                  maximum=1,
                                  step=0.05,
                                  value=0.5)
            run_button = gr.Button('Run')
        with gr.Column():
            result = gr.Image(label='Masked')

    inputs = [image, threshold]
    gr.Examples(examples=examples,
                inputs=inputs,
                outputs=result,
                fn=fn,
                cache_examples=os.getenv('CACHE_EXAMPLES') == '1')
    run_button.click(fn=fn, inputs=inputs, outputs=result, api_name='predict')
demo.queue(max_size=15).launch()
