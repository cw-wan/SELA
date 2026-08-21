import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    model_name: str
    api_type: str
    api_key: str
    deployment_name: str
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    max_completion_tokens: int = 15_000
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = None
    extra_body: Optional[dict] = None
    seed: Optional[int] = None


def load_llm_config(config_path: str, model_name: str) -> LLMConfig:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if model_name not in cfg:
        raise ValueError(f"Model '{model_name}' not found in {config_path}")

    mc = cfg[model_name]

    if "azure_endpoint" in mc:
        api_type = "azure_openai"
        base_url = mc["azure_endpoint"]
    elif "base_url" in mc:
        api_type = "openai"
        base_url = mc["base_url"]
    else:
        api_type = "google"
        base_url = None

    return LLMConfig(
        model_name=model_name,
        api_type=api_type,
        api_key=mc["api_key"],
        deployment_name=mc.get("deployment_name", model_name),
        base_url=base_url,
        api_version=mc.get("api_version"),
        max_completion_tokens=mc.get("max_completion_tokens", 15_000),
        temperature=mc.get("temperature"),
        reasoning_effort=mc.get("reasoning_effort"),
        extra_body=mc.get("extra_body"),
        seed=mc.get("seed"),
    )
