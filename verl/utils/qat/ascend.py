# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

from verl.utils.qat.compressed_tensors import _HIGH_PRECISION_DTYPES, _PACKED_EXPERTS, _unpack_experts

logger = logging.getLogger(__name__)


def detect_ascend_modelslim(model_path: str) -> bool:
    try:
        from vllm_ascend.quantization.utils import detect_quantization_method
    except ImportError:
        return False
    return detect_quantization_method(model_path) == "ascend"


_EXPERT_INDEX_RE = re.compile(r"(\.experts\.)\d+(\.)")


def _modelslim_template_key(name: str) -> str:
    """Fold an expert index in a weight name to the description's template key.

    ModelSlim descriptions store one template entry per layer (e.g.
    ``...mlp.experts.0.gate_proj.weight``); vllm-ascend expands that entry to
    every expert when it creates the engine parameters (modelslim_config
    normalizes expert indices, vllm_ascend/quantization/configs/
    modelslim_config.py:669-672). The export side must match that template
    semantics, otherwise only expert 0 is quantized and the engine's scales
    for the remaining experts stay at their ``torch.empty`` init. Literal
    entries in the description still win over the template fallback.
    """
    return _EXPERT_INDEX_RE.sub(r"\g<1>0\g<2>", name, count=1)


class AscendMxfp8WeightQuantizer:
    """Export trainer weights in vLLM-Ascend's native W8A8_MXFP8 layout."""

    def __init__(self, quant_description: Mapping[str, Any]):
        self.quant_description = dict(quant_description)
        self._l0_quant = 0
        self._l0_pass = 0
        unsupported = sorted(
            {
                quant_type
                for name, quant_type in self.quant_description.items()
                if name.endswith(".weight") and quant_type not in {"W8A8_MXFP8", "FLOAT"}
            }
        )
        if unsupported:
            raise NotImplementedError(f"Unsupported Ascend ModelSlim weight quantization types: {unsupported}")

    @classmethod
    def from_model_path(cls, model_path: str):
        with (Path(model_path) / "quant_model_description.json").open() as f:
            return cls(json.load(f))

    @staticmethod
    def _export_weight(name: str, weight: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        import torch_npu

        quantized, scale = torch_npu.npu_dynamic_mx_quant(weight, axis=-1, dst_type=torch.float8_e4m3fn)
        groups = (weight.shape[-1] + 31) // 32
        yield name, quantized
        yield name + "_scale", scale.flatten(-2, -1)[..., :groups]

    def __call__(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterator[tuple[str, torch.Tensor]]:
        self._l0_quant = 0
        self._l0_pass = 0
        try:
            for name, tensor in weights:
                packed = _PACKED_EXPERTS.match(name) if tensor.ndim == 3 else None
                if packed:
                    for weight_name, weight in _unpack_experts(name, tensor):
                        yield from self._export_selected_weight(weight_name, weight)
                else:
                    yield from self._export_selected_weight(name, tensor)
        finally:
            logger.info("[L0-quant-export] quantized=%d passthrough=%d", self._l0_quant, self._l0_pass)

    def _export_selected_weight(self, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        quant_type = self.quant_description.get(name)
        if quant_type is None and ".experts." in name:
            # Path A (20261105): literal miss -> fall back to the experts.0
            # template entry, matching the engine's parameter-creation
            # semantics (see _modelslim_template_key).
            quant_type = self.quant_description.get(_modelslim_template_key(name))
        if (
            quant_type != "W8A8_MXFP8"
            or tensor.ndim != 2
            or tensor.dtype not in _HIGH_PRECISION_DTYPES
        ):
            self._l0_pass += 1
            yield name, tensor
            return
        self._l0_quant += 1
        yield from self._export_weight(name, tensor)
