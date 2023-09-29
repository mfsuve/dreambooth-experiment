from pathlib import Path
from typing import Dict
import random

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DreamBoothDataset(Dataset):
    def __init__(self, config: Dict, tokenizer):
        self.tokenizer = tokenizer
        size = config["image_size"]

        self.instance_images_path = list(Path(config["paths"]["instance_images"]).iterdir())
        self.class_images_path = list(Path(config["paths"]["class_images"]).iterdir())

        self.instance_prompt = config["prompts"]["instance"]
        self.class_prompt = config["prompts"]["class"]

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self.length = max(self.num_class_images, self.num_instance_images)

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.length

    def process_image_prompt_pair(self, image: Image, prompt: str):
        if not image.mode == "RGB":
            image = image.convert("RGB")

        return {
            "instance_images": self.image_transforms(image),
            "instance_prompt_ids": self.tokenizer(
                prompt, truncation=True, max_length=self.tokenizer.model_max_length
            ).input_ids,
        }

    def __getitem__(self, index):
        return {
            **self.process_image_prompt_pair(
                Image.open(self.instance_images_path[index % self.num_instance_images]), self.instance_prompt
            ),
            **self.process_image_prompt_pair(
                Image.open(self.class_images_path[index % self.num_class_images]), self.class_prompt
            ),
        }
