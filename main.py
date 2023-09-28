from pathlib import Path
from typing import Dict
import yaml
import hashlib
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from diffusers import AutoencoderKL, StableDiffusionPipeline


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return {
            "prompt": self.prompt,
            "index": index,
        }


def generate_class_images(accelerator: Accelerator, config: Dict):
    class_images_path: Path = Path(config["paths"]["class_images"])
    class_images_path.mkdir(parents=True, exist_ok=True)
    current_num_class_images = len(list(class_images_path.iterdir()))
    num_new_images = config["hyperparams"]["prior_sampling"]["num_class_images"] - current_num_class_images

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

        sample_dataloader = DataLoader(
            PromptDataset(config["prompts"]["class"], num_new_images),
            batch_size=config["hyperparams"]["prior_sampling"]["sample_batch_size"],
        )
        sample_dataloader = accelerator.prepare(sample_dataloader)

        with torch.autocast("cuda"), torch.inference_mode():
            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    image_hash = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = (
                        class_images_path / f"{current_num_class_images + example['index'][i]}-{image_hash}.jpg"
                    )
                    image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(config):
    accelerator = Accelerator()
    generate_class_images(accelerator, config)


if __name__ == "__main__":
    with open("config.yml", "r") as config_file:
        config = yaml.full_load(config_file)
    main(config)
