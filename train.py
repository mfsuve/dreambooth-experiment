from typing import Dict
from itertools import chain

from data import DreamBoothDataset

import torch
from bitsandbytes.optim import AdamW8bit
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


class Trainer:
    def __init__(self, config: Dict):
        pretrained_model_name = config["pretrained_model_name"]
        hyperparams = config["hyperparams"]

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
        self.vae.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")

        self.vae.enable_xformers_memory_efficient_attention()
        self.unet.enable_xformers_memory_efficient_attention()

        if hyperparams.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()

        self.optimizer = AdamW8bit(
            chain(self.unet.parameters(), self.text_encoder.parameters()),
            lr=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"],
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

        self.train_dataloader(
            DreamBoothDataset(config, self.tokenizer),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, examples):
        input_ids = [e["instance_prompt_ids"] for e in examples] + [e["class_prompt_ids"] for e in examples]
        pixel_values = [e["instance_images"] for e in examples] + [e["class_images"] for e in examples]

        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
