# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Dependency-light Diffusers-to-Cosmos weight key mapping."""

from __future__ import annotations

import re

_DIFFUSERS_ROOT_INDEX = "model.safetensors.index.json"
_DIFFUSERS_MODEL_INDEX = "model_index.json"
_DIFFUSERS_DROP_WEIGHT_PATH_RES: tuple[re.Pattern[str], ...] = (re.compile(r"^(?!transformer/|vision_encoder/)"),)
_DIFFUSERS_DROP_KEY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:feature_extractor|image_processor|scheduler|sound_tokenizer|text_encoder|tokenizer|vae)\."),
)
_DIFFUSERS_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^vision_encoder\."), ""),
    (re.compile(r"^model\.net\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^model\.lm_head\."), "language_model.lm_head."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (re.compile(r"^model\.visual\."), "language_model.visual."),
    (re.compile(r"^visual\."), "language_model.visual."),
    (
        re.compile(r"^(blocks\.|deepstack_merger_list\.|merger\.|patch_embed\.|pos_embed\.)(.*)$"),
        r"language_model.visual.\1\2",
    ),
    (
        re.compile(r"^language_model\.(?!model\.|lm_head\.|visual\.)(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^model\.(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)
_DIFFUSERS_NET_KEY_PREFIXES: tuple[str, ...] = (
    "action2llm.",
    "action_pos_embed.",
    "language_model.",
    "latent_pos_embed.",
    "llm2action.",
    "llm2sound.",
    "llm2vae.",
    "sound2llm.",
    "time_embedder.",
    "vae2llm.",
)
_DIFFUSERS_NET_KEYS: frozenset[str] = frozenset(
    {
        "action_modality_embed",
        "latent_pos_embed",
        "sound_modality_embed",
    }
)


def _should_drop_diffusers_weight_path(path: str) -> bool:
    path = path.replace("\\", "/")
    return bool(path) and any(pattern.search(path) is not None for pattern in _DIFFUSERS_DROP_WEIGHT_PATH_RES)


def _should_drop_diffusers_key(name: str) -> bool:
    return any(pattern.search(name) is not None for pattern in _DIFFUSERS_DROP_KEY_RES)


def _is_diffusers_model_weight_path(path: str) -> bool:
    return bool(path) and not _should_drop_diffusers_weight_path(path)


def _apply_diffusers_key_mapping(name: str) -> str:
    for pattern, replacement in _DIFFUSERS_KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


def _is_loadable_diffusers_net_key(name: str) -> bool:
    return name in _DIFFUSERS_NET_KEYS or name.startswith(_DIFFUSERS_NET_KEY_PREFIXES)


def _diffusers_to_net_key(name: str, weight_path: str = "") -> str | None:
    """Rename a Diffusers checkpoint key to its OmniMoTModel.net subtree key."""

    if _should_drop_diffusers_weight_path(weight_path) or _should_drop_diffusers_key(name):
        return None
    net_key = _apply_diffusers_key_mapping(name)
    if _should_drop_diffusers_key(net_key) or not _is_loadable_diffusers_net_key(net_key):
        return None
    return net_key
