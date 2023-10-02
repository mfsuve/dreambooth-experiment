from typing import Dict
from itertools import chain
from tqdm.auto import tqdm

from data import DreamBoothDataset

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from bitsandbytes.optim import AdamW8bit
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel


class Trainer:
    def __init__(self, config: Dict):
        pretrained_model_name = config["pretrained_model_name"]
        self.hyperparams = config["hyperparams"]

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name, subfolder="text_encoder")

        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name, subfolder="vae")
        self.vae.requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name, subfolder="unet")

        self.vae.enable_xformers_memory_efficient_attention()
        self.unet.enable_xformers_memory_efficient_attention()

        if self.hyperparams.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()

        self.optimizer = AdamW8bit(
            chain(self.unet.parameters(), self.text_encoder.parameters()),
            lr=self.hyperparams["learning_rate"],
            weight_decay=self.hyperparams["weight_decay"],
        )

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name, subfolder="scheduler")

        self.train_dataloader = DataLoader(
            DreamBoothDataset(config, self.tokenizer),
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def collate_fn(self, examples):
        input_ids = [e["instance_prompt_ids"] for e in examples] + [e["class_prompt_ids"] for e in examples]
        pixel_values = [e["instance_images"] for e in examples] + [e["class_images"] for e in examples]

        input_ids = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

        return {"input_ids": input_ids, "pixel_values": pixel_values, "size": pixel_values.shape[0]}

    def calculate_loss_on_batch(self, batch, dtype):
        with torch.no_grad():
            latents = self.vae.encode(batch["pixel_values"].to(dtype=dtype)).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
        ).long()
        latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]
        model_pred = self.unet(latents, timesteps, encoder_hidden_states).sample

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
        target, target_prior = torch.chunk(target, 2, dim=0)

        instance_loss = F.mse_loss(model_pred.float(), target.float())
        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float())
        return instance_loss + self.hyperparams["prior_loss_weight"] * prior_loss

    def train_one_epoch(self, accelerator, dtype):
        self.unet.train()
        self.text_encoder.train()
        total_loss = 0
        for batch in self.train_dataloader:
            with accelerator.accumulate(self.unet):
                loss = self.calculate_loss_on_batch(batch, dtype)
                accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                total_loss += loss.detach().item() * batch["size"]
        return total_loss / len(self.train_dataloader)

    def train(self, accelerator):
        dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32

        self.vae.to(accelerator.device, dtype=dtype)
        self.unet, self.text_encoder, self.optimizer, self.train_dataloader = accelerator.prepare(
            self.unet, self.text_encoder, self.optimizer, self.train_dataloader
        )

        num_epochs = self.hyperparams["num_epochs"]
        for epoch in tqdm(range(num_epochs), disable=not accelerator.is_local_main_process, desc="Epochs"):
            print(f"\nEpoch [{epoch + 1:>{len(str(num_epochs))}}/{num_epochs}]")
            avg_loss = self.train_one_epoch(accelerator, dtype)
            print(f"\tLoss: {avg_loss:.4f}")
