from pathlib import Path
from typing import Dict
import yaml
import hashlib
from tqdm.auto import tqdm

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, StableDiffusionPipeline

from train import Trainer


def generate_class_images(accelerator: Accelerator, config: Dict):
    class_images_path: Path = Path(config["train"]["paths"]["class_images"])
    class_images_path.mkdir(parents=True, exist_ok=True)
    current_num_class_images = len(list(class_images_path.iterdir()))
    num_new_images = config["train"]["hyperparams"]["num_class_images"] - current_num_class_images

    if num_new_images > 0:
        torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
        pipeline = StableDiffusionPipeline.from_pretrained(
            config["pretrained_model_name"],
            vae=AutoencoderKL.from_pretrained(
                config["pretrained_model_name"], subfolder="vae", torch_dtype=torch_dtype
            ),
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(accelerator.device)

        with torch.autocast("cuda"), torch.inference_mode():
            for index in tqdm(range(num_new_images), desc="Generating class images"):
                image = pipeline(config["train"]["prompts"]["class"]).images[0]
                image_hash = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_path / f"{current_num_class_images + index}-{image_hash}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(config):
    accelerator = Accelerator()
    generate_class_images(accelerator, config)
    trainer = Trainer(config)
    trainer.train(accelerator)


if __name__ == "__main__":
    with open("config.yml", "r") as config_file:
        config = yaml.full_load(config_file)
    main(config)
