# Copyright 2026 The HuggingFace Team & 8b-is. All rights reserved.
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
    from ..utils.quantization_config import NvidiaFp8TmaConfig

from ..utils import is_torch_available, logging
from ..utils.import_utils import is_fp8_supported, is_hopper_available


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class NvidiaFp8TmaHfQuantizer(HfQuantizer):
    """
    NVIDIA FP8 and Hopper/Blackwell TMA (Tensor Memory Accelerator) Quantizer:
    Converts linear layers into hardware-accelerated `NvidiaFp8Linear` layers with 128-byte TMA descriptor support.
    """

    requires_calibration = False
    quantization_config: "NvidiaFp8TmaConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            logger.warning_once(
                "NVIDIA FP8/TMA quantization requested without CUDA GPU. Fallback simulation mode will be used."
            )
        elif not is_fp8_supported():
            logger.warning_once(
                "NVIDIA GPU does not natively support FP8 Tensor Cores (requires Ada CC 8.9+, Hopper CC 9.0+, or Blackwell CC 10.0+). Emulation mode active."
            )

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations.nvidia_fp8_tma import replace_with_nvidia_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        fp8_dtype = (
            torch.float8_e4m3fn if self.quantization_config.fp8_format == "e4m3" else torch.float8_e5m2
        )

        model = replace_with_nvidia_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            fp8_dtype=fp8_dtype,
            use_tma=self.quantization_config.use_tma,
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        return False
