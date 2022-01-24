#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import subprocess
import sys
import tarfile
from typing import Callable

# workaround for https://github.com/gradio-app/gradio/issues/483
command = 'pip install -U gradio==2.7.0'
subprocess.call(command.split())

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

TOKEN = os.environ['TOKEN']

MODEL_REPO = 'hysts/bizarre-pose-estimator-models'
MODEL_FILENAME = 'segmenter.pth'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--score-slider-step', type=float, default=0.05)
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    parser.add_argument('--allow-screenshot', action='store_true')
    return parser.parse_args()


def load_sample_image_paths() -> list[pathlib.Path]:
    image_dir = pathlib.Path('images')
    if not image_dir.exists():
        dataset_repo = 'hysts/sample-images-TADNE'
        path = huggingface_hub.hf_hub_download(dataset_repo,
                                               'images.tar.gz',
                                               repo_type='dataset',
                                               use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()
    return sorted(image_dir.glob('*'))


def load_model(
        device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    path = huggingface_hub.hf_hub_download(MODEL_REPO,
                                           MODEL_FILENAME,
                                           use_auth_token=TOKEN)
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

    mask = np.asarray(probs)
    mask[mask < score_threshold] = 0
    mask[mask > 0] = 1
    mask = mask.astype(bool)

    res = np.asarray(image)
    res[~mask] = 255
    return res


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    image_paths = load_sample_image_paths()
    examples = [[path.as_posix(), args.score_threshold]
                for path in image_paths]

    model, final_head = load_model(device)
    transform = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    func = functools.partial(predict,
                             transform=transform,
                             device=device,
                             model=model,
                             final_head=final_head)
    func = functools.update_wrapper(func, predict)

    repo_url = 'https://github.com/ShuhongChen/bizarre-pose-estimator'
    title = 'ShuhongChen/bizarre-pose-estimator (segmenter)'
    description = f'A demo for {repo_url}'
    article = None

    gr.Interface(
        func,
        [
            gr.inputs.Image(type='pil', label='Input'),
            gr.inputs.Slider(0,
                             1,
                             step=args.score_slider_step,
                             default=args.score_threshold,
                             label='Score Threshold'),
        ],
        gr.outputs.Image(label='Masked'),
        theme=args.theme,
        title=title,
        description=description,
        article=article,
        examples=examples,
        allow_screenshot=args.allow_screenshot,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
