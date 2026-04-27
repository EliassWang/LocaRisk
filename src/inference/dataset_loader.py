import json
import os
from datasets import load_dataset


def load_dataset_data(name: str, config_path: str = "configs/datasets.json"):
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)

    item = configs[name]

    ds = load_dataset(
        path=item.get("dataset_name"),
        name=item.get("dataset_config"),
        split=item.get("split"),
        streaming=item.get("streaming", False),
        cache_dir=item.get("cache_dir")
    )

    return ds
