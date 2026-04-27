import logging
from pathlib import Path
import torch

from src.evaluation.cost import add_input_tokens, empty_cost_summary


def infer_repo_id_from_local_path(model_path: str | Path) -> str:
    model_dir_name = Path(model_path).name
    if "_" not in model_dir_name:
        raise ValueError(
            f"Cannot infer Hugging Face repo id from model path: {model_path}"
        )

    org_name, repo_name = model_dir_name.split("_", 1)
    return f"{org_name}/{repo_name}"


def has_local_model_files(model_path: str | Path) -> bool:
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    tokenizer_artifacts = (
        model_path / "tokenizer_config.json",
        model_path / "tokenizer.json",
        model_path / "tokenizer.model",
    )
    return config_path.exists() and any(path.exists() for path in tokenizer_artifacts)


def ensure_local_model_path(model_path: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    model_path = Path(model_path)
    if has_local_model_files(model_path):
        return model_path

    model_path.mkdir(parents=True, exist_ok=True)
    repo_id = infer_repo_id_from_local_path(model_path)
    logging.info("Downloading model %s into %s", repo_id, model_path)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return model_path


def load_local_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    local_model_path = ensure_local_model_path(model_path)
    return AutoTokenizer.from_pretrained(
        str(local_model_path),
        local_files_only=True,
        trust_remote_code=True
    )


class LLM:
    def __init__(
            self,
            model_path: str,
            do_sample: None,
            temperature: None,
            max_new_tokens: None,
    ):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        self.config = {
            "model_path": model_path,
            "do_sample": do_sample,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "enable_thinking": False
        }
        self.cost = empty_cost_summary()
        local_model_path = ensure_local_model_path(self.config["model_path"])
        self.config["model_path"] = str(local_model_path)

        self.tok = load_local_tokenizer(self.config["model_path"])

        base = {
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": True
        }

        try:
            q_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_path"],
                quantization_config=q_cfg,
                **base
            ).eval()
        except Exception as e:
            logging.warning("4-bit fail, using fp16: %s", e)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config["model_path"],
                torch_dtype=torch.float16,
                **base
            ).eval()

    @torch.no_grad()
    def __call__(self, prompt: str, task: str = None, **kwargs) -> str:
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": f"{task}:\n{prompt}" if task else prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        add_input_tokens(self.cost, input_len)

        # Determine if we are sampling or using greedy decoding
        do_sample = kwargs.get("do_sample", self.config["do_sample"])

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.config["max_new_tokens"]),
            "do_sample": do_sample,
            "pad_token_id": self.tok.eos_token_id,
        }

        if do_sample:
            # Only pass sampling parameters if do_sample is True
            gen_kwargs["temperature"] = kwargs.get("temperature", self.config["temperature"])
            gen_kwargs["top_p"] = kwargs.get("top_p", 1.0)
            gen_kwargs["top_k"] = kwargs.get("top_k", 50)
        else:
            # Explicitly set to None to suppress warnings when do_sample=False
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        outputs = self.model.generate(**inputs, **gen_kwargs)
        response = self.tok.decode(outputs[0][input_len:], skip_special_tokens=True)

        return response


__all__ = [
    "LLM",
    "ensure_local_model_path",
    "has_local_model_files",
    "infer_repo_id_from_local_path",
    "load_local_tokenizer",
]
