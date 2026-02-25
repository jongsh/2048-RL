import argparse
import os
import re
from typing import Dict, List, Tuple

import torch
from omegaconf import OmegaConf

from configs.config import Configuration
from models.mlp import MLPPolicy, MLPValue
from models.resnet import ResNetPolicy, ResNetValue
from models.transformer import TransformerEncoderPolicy, TransformerEncoderValue


def _build_model(model_name: str, task: str, config: Configuration) -> torch.nn.Module:
    model_name = model_name.lower()
    task = task.lower()
    if model_name == "mlp":
        return MLPPolicy(config) if task == "policy" else MLPValue(config)
    if model_name == "resnet":
        return ResNetPolicy(config) if task == "policy" else ResNetValue(config)
    if model_name == "transformer":
        return TransformerEncoderPolicy(config) if task == "policy" else TransformerEncoderValue(config)
    raise ValueError(f"Unsupported model '{model_name}'. Use one of: mlp, resnet, transformer.")


def _infer_head_keys(model_name: str, state_dict: Dict[str, torch.Tensor]) -> List[str]:
    model_name = model_name.lower()
    if model_name == "resnet":
        return [k for k in state_dict.keys() if k.startswith("fc.")]
    if model_name == "transformer":
        return [k for k in state_dict.keys() if k.startswith("output_layer.")]
    if model_name == "mlp":
        linear_indices = []
        pattern = re.compile(r"^network\.network\.(\d+)\.weight$")
        for key in state_dict.keys():
            m = pattern.match(key)
            if m:
                linear_indices.append(int(m.group(1)))
        if not linear_indices:
            raise RuntimeError("Cannot infer MLP head layer from state dict keys.")
        last_idx = max(linear_indices)
        prefix = f"network.network.{last_idx}."
        return [k for k in state_dict.keys() if k.startswith(prefix)]
    raise ValueError(f"Unsupported model '{model_name}'.")


def _resolve_weight_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        candidates = ["model.pth", "q_network.pth", "target_network.pth"]
        for name in candidates:
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"Cannot find weight file at '{path}'. If using a directory, it must include model.pth/q_network.pth/target_network.pth."
    )


def _default_src_weight_name(src_task: str) -> str:
    return "model.pth" if src_task == "policy" else "q_network.pth"


def _default_dst_weight_name(dst_task: str) -> str:
    return "model.pth" if dst_task == "policy" else "q_network.pth"


def _merge_backbone(
    src_state: Dict[str, torch.Tensor],
    dst_state: Dict[str, torch.Tensor],
    head_keys: List[str],
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    merged = {k: v.clone() for k, v in dst_state.items()}
    copied = []
    skipped = []
    head_key_set = set(head_keys)

    for key, dst_tensor in dst_state.items():
        if key in head_key_set:
            skipped.append(key)
            continue
        src_tensor = src_state.get(key)
        if src_tensor is None:
            skipped.append(key)
            continue
        if src_tensor.shape != dst_tensor.shape:
            skipped.append(key)
            continue
        merged[key] = src_tensor.clone()
        copied.append(key)
    return merged, copied, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert policy/value weights by reusing backbone and reinitializing final head layer."
    )
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "resnet", "transformer"])
    parser.add_argument("--src-task", type=str, required=True, choices=["policy", "value"])
    parser.add_argument("--dst-task", type=str, required=True, choices=["policy", "value"])
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory that contains config.yaml and source weight file.",
    )
    parser.add_argument(
        "--src-weight-name",
        type=str,
        default=None,
        help="Source weight filename inside checkpoint dir. Default: model.pth(policy) or q_network.pth(value).",
    )
    parser.add_argument(
        "--dst-weight-name",
        type=str,
        default=None,
        help="Output weight filename. Default: model.pth(policy) or q_network.pth(value).",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for converted checkpoint.")
    args = parser.parse_args()

    ckpt_dir = args.checkpoint_dir
    config_path = os.path.join(ckpt_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Cannot find config at '{config_path}'.")

    config = Configuration(config_path=config_path, from_scratch=False)
    src_model = _build_model(args.model, args.src_task, config)
    dst_model = _build_model(args.model, args.dst_task, config)

    src_weight_name = args.src_weight_name or _default_src_weight_name(args.src_task)
    src_weight_path = _resolve_weight_path(os.path.join(ckpt_dir, src_weight_name))
    src_state = torch.load(src_weight_path, map_location="cpu", weights_only=True)
    src_model.load_state_dict(src_state, strict=True)

    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    head_keys = _infer_head_keys(args.model, dst_state)
    merged_state, copied, skipped = _merge_backbone(src_state, dst_state, head_keys)
    dst_model.load_state_dict(merged_state, strict=True)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    dst_weight_name = args.dst_weight_name or _default_dst_weight_name(args.dst_task)
    out_weight_path = os.path.join(out_dir, dst_weight_name)
    torch.save(dst_model.state_dict(), out_weight_path)
    OmegaConf.save(config.config, os.path.join(out_dir, "config.yaml"))

    print(f"[OK] Source: {src_weight_path}")
    print(f"[OK] Checkpoint Config: {config_path}")
    print(f"[OK] Target: {args.model} {args.dst_task}")
    print(f"[OK] Output Dir: {out_dir}")
    print(f"[OK] Output Weight: {out_weight_path}")
    print(f"[INFO] Copied params: {len(copied)}")
    print(f"[INFO] Reinitialized/Skipped params: {len(skipped)}")
    print(f"[INFO] Head params kept from target init: {head_keys}")


if __name__ == "__main__":
    main()
