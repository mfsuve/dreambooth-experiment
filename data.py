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

        self.instance_images_path = list(Path(config['train']["paths"]["instance_images"]).iterdir())
        self.class_images_path = list(Path(config['train']["paths"]["class_images"]).iterdir())

        self.instance_prompt_ids = self.tokenize(config['train']["prompts"]["instance"])
        self.class_prompt_ids = self.tokenize(config['train']["prompts"]["class"])

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

    def tokenize(self, prompt: str):
        return self.tokenizer(prompt, truncation=True, max_length=self.tokenizer.model_max_length).input_ids

    def load_image_from_path(self, path: Path):
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        return self.image_transforms(image)

    def __getitem__(self, index):
        return {
            "instance_images": self.load_image_from_path(self.instance_images_path[index % self.num_instance_images]),
            "instance_prompt_ids": self.instance_prompt_ids,
            "class_images": self.load_image_from_path(self.class_images_path[index % self.num_class_images]),
            "class_prompt_ids": self.class_prompt_ids,
        }
