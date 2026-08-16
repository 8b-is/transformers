# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import BitNetMlxQuantConfig

from ..utils import is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class BitNetMlxHfQuantizer(HfQuantizer):
    """
    1.58-bit ternary quantization from MLX-QUANT (BitNet b1.58 format):
    Before loading: it converts the linear layers into BitNetTernaryLinear layers during loading.
    """

    requires_calibration = False
    quantization_config: "BitNetMlxQuantConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not torch.cuda.is_available() and not is_mps:
            logger.warning_once(
                "You don't have a CUDA or Apple Silicon (MPS) GPU available. BitNet MLX ternary layers will run on CPU."
            )

        device_map = kwargs.get("device_map")
        if device_map is None and torch.cuda.is_available():
            pass

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations.bitnet_mlx import replace_with_bitnet_mlx_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_bitnet_mlx_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    def get_weight_conversions(self):
        return []
