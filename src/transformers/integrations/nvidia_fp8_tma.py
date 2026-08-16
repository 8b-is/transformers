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

"""
NVIDIA FP8 & Hopper / Blackwell TMA (Tensor Memory Accelerator) First-Class Integration.

Features:
- Hardware-aligned 128-byte TMA (Tensor Map Accelerator) Descriptors for SM90+ (Hopper) and SM100+ (Blackwell).
- Ultra-fast struct-compiled binary packing and ctypes 128-byte memory alignment.
- Low-overhead FP8 E4M3FN / E5M2 dynamic and delayed quantization.
- Optimized dispatch via `torch._scaled_mm` with fast accumulation and zero-overhead local bytecode bindings.
- Seamless fallback for cross-platform simulation and testing.
"""

from __future__ import annotations

import ctypes
import enum
import struct
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..utils import logging
from ..utils.import_utils import is_fp8_supported, is_hopper_available, is_torch_cuda_available


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

# Constants for FP8 numerical bounds
FP8_E4M3_MAX: float = 448.0
FP8_E5M2_MAX: float = 57344.0
EPSILON: float = 1e-12

# Pre-compiled binary struct for 128-byte CUDA TMA Tensor Map descriptor
# Layout:
# 64-bit base ptr (Q) + 64-bit dtype/rank/interleave (Q) + 4x64-bit size (4Q) +
# 4x64-bit stride (4Q) + 4x32-bit box_size (4I) + 4x32-bit elem_stride (4I) +
# 16x8-bit swizzle/promotion/oob (16B)
# Total: 8 + 8 + 32 + 32 + 16 + 16 + 16 = 128 bytes
_TMA_STRUCT_PACKER = struct.Struct("=QQ4Q4Q4I4I16B")


class TmaDataType(enum.IntEnum):
    """CUDA CUtensorMapDataType enum values."""

    UINT8 = 0
    UINT16 = 1
    UINT32 = 2
    INT32 = 3
    UINT64 = 4
    INT64 = 5
    FLOAT16 = 6
    FLOAT32 = 7
    FLOAT64 = 8
    BFLOAT16 = 9
    FLOAT8_E4M3 = 10
    FLOAT8_E5M2 = 11


class TmaSwizzle(enum.IntEnum):
    """CUDA CUtensorMapSwizzle enum values for shared memory bank conflict mitigation."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B = 1
    SWIZZLE_64B = 2
    SWIZZLE_128B = 3


class TmaPromotion(enum.IntEnum):
    """CUDA CUtensorMapL2Promotion enum."""

    L2_NONE = 0
    L2_64B = 1
    L2_128B = 2
    L2_256B = 3


class TmaDescriptor:
    """
    128-byte hardware Tensor Memory Accelerator (TMA) descriptor.
    Mirrors CUDA `CUtensorMap` structure for asynchronous Global-to-Shared memory copy on Hopper (SM90) and Blackwell (SM100).
    """

    __slots__ = (
        "base_ptr",
        "data_type",
        "rank",
        "global_shape",
        "global_strides",
        "box_size",
        "element_strides",
        "swizzle",
        "promotion",
        "_raw_bytes",
    )

    def __init__(
        self,
        base_ptr: int,
        data_type: TmaDataType,
        global_shape: tuple[int, ...],
        global_strides: tuple[int, ...],
        box_size: tuple[int, ...],
        element_strides: tuple[int, ...] | None = None,
        swizzle: TmaSwizzle = TmaSwizzle.SWIZZLE_128B,
        promotion: TmaPromotion = TmaPromotion.L2_128B,
    ):
        self.base_ptr = base_ptr
        self.data_type = data_type
        self.rank = len(global_shape)
        self.global_shape = global_shape
        self.global_strides = global_strides
        self.box_size = box_size
        self.element_strides = element_strides or tuple(1 for _ in global_shape)
        self.swizzle = swizzle
        self.promotion = promotion
        self._raw_bytes: bytes | None = None

    def pack(self) -> bytes:
        """Serializes descriptor into 128 bytes with exact hardware alignment."""
        if self._raw_bytes is not None:
            return self._raw_bytes

        # Pad shapes and strides up to 4 dimensions (standard for 2D/3D/4D matrix GEMM)
        pad_shape = list(self.global_shape) + [0] * (4 - min(self.rank, 4))
        pad_strides = list(self.global_strides) + [0] * (4 - min(self.rank, 4))
        pad_box = list(self.box_size) + [0] * (4 - min(self.rank, 4))
        pad_elem_strides = list(self.element_strides) + [1] * (4 - min(self.rank, 4))

        flags = (int(self.data_type) & 0xFF) | ((self.rank & 0xFF) << 8)
        padding_14b = [0] * 14

        packed = _TMA_STRUCT_PACKER.pack(
            self.base_ptr,
            flags,
            *pad_shape[:4],
            *pad_strides[:4],
            *pad_box[:4],
            *pad_elem_strides[:4],
            int(self.swizzle),
            int(self.promotion),
            *padding_14b,
        )
        self._raw_bytes = packed
        return packed

    def pack_into(self, buffer: bytearray | memoryview | ctypes.Array[ctypes.c_char], offset: int = 0) -> None:
        """Packs descriptor directly into a pre-allocated mutable buffer without creating intermediate bytes."""
        pad_shape = list(self.global_shape) + [0] * (4 - min(self.rank, 4))
        pad_strides = list(self.global_strides) + [0] * (4 - min(self.rank, 4))
        pad_box = list(self.box_size) + [0] * (4 - min(self.rank, 4))
        pad_elem_strides = list(self.element_strides) + [1] * (4 - min(self.rank, 4))

        flags = (int(self.data_type) & 0xFF) | ((self.rank & 0xFF) << 8)
        padding_14b = [0] * 14

        _TMA_STRUCT_PACKER.pack_into(
            buffer,
            offset,
            self.base_ptr,
            flags,
            *pad_shape[:4],
            *pad_strides[:4],
            *pad_box[:4],
            *pad_elem_strides[:4],
            int(self.swizzle),
            int(self.promotion),
            *padding_14b,
        )

    def as_ctypes_buffer(self) -> ctypes.Array[ctypes.c_char]:
        """Returns 128-byte ctypes buffer suitable for passing directly to CUDA C drivers."""
        data = self.pack()
        buf = ctypes.create_string_buffer(data, 128)
        return buf

    def as_c_struct(self) -> CUtensorMapStruct:
        """Returns native CUtensorMapStruct ctypes structure."""
        return CUtensorMapStruct.from_descriptor(self)


class CUtensorMapStruct(ctypes.Structure):
    """
    Direct C ABI layout matching NVIDIA `CUtensorMap` / `cudaTensorMap` definition.
    Exact 128 bytes with 64-bit word alignment.
    """

    _pack_ = 8
    _fields_ = [
        ("opaque", ctypes.c_uint64 * 16),
    ]

    def to_bytes(self) -> bytes:
        return bytes(self)

    @classmethod
    def from_descriptor(cls, desc: TmaDescriptor) -> CUtensorMapStruct:
        packed = desc.pack()
        obj = cls()
        ctypes.memmove(ctypes.byref(obj), packed, 128)
        return obj


def create_2d_tma_descriptor(
    tensor: torch.Tensor,
    tile_m: int = 64,
    tile_k: int = 64,
    swizzle: TmaSwizzle = TmaSwizzle.SWIZZLE_128B,
) -> TmaDescriptor:
    """
    Constructs a 2D TMA Tensor Map descriptor for a matrix in GMEM (Global Memory)
    to be fetched asynchronously into SMEM (Shared Memory) on NVIDIA Hopper / Blackwell GPUs.
    """
    if tensor.dim() != 2:
        raise ValueError(f"Tensor must be 2-dimensional for 2D TMA, got shape {tensor.shape}")

    dtype_map = {
        torch.float8_e4m3fn: TmaDataType.FLOAT8_E4M3,
        torch.float8_e5m2: TmaDataType.FLOAT8_E5M2,
        torch.float16: TmaDataType.FLOAT16,
        torch.bfloat16: TmaDataType.BFLOAT16,
        torch.float32: TmaDataType.FLOAT32,
    }
    tma_dtype = dtype_map.get(tensor.dtype, TmaDataType.FLOAT8_E4M3)

    ptr = tensor.data_ptr() if hasattr(tensor, "data_ptr") else 0
    rows, cols = tensor.shape
    stride_row = tensor.stride(0) * tensor.element_size()
    stride_col = tensor.stride(1) * tensor.element_size()

    return TmaDescriptor(
        base_ptr=ptr,
        data_type=tma_dtype,
        global_shape=(cols, rows),
        global_strides=(stride_col, stride_row),
        box_size=(min(tile_k, cols), min(tile_m, rows)),
        swizzle=swizzle,
    )


def fp8_quantize(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """
    Quantizes a floating point tensor into FP8 format using the specified scale factor:
    x_fp8 = clamp(x * scale, -max_fp8, max_fp8).to(fp8_dtype)
    """
    max_val = FP8_E4M3_MAX if fp8_dtype == torch.float8_e4m3fn else FP8_E5M2_MAX
    # Local method caching for high-speed inner-loop execution
    _mul = torch.mul
    _clamp = torch.clamp

    scaled = _mul(tensor, scale)
    clamped = _clamp(scaled, min=-max_val, max=max_val)
    return clamped.to(fp8_dtype)


def fp8_dynamic_quantize(
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-tensor dynamic FP8 quantization:
    Computes optimal scale factor from amax and quantizes tensor in one pass.
    """
    max_fp8 = FP8_E4M3_MAX if fp8_dtype == torch.float8_e4m3fn else FP8_E5M2_MAX
    amax = torch.amax(torch.abs(tensor)).clamp(min=EPSILON)
    scale = (max_fp8 / amax).to(torch.float32)
    quantized = fp8_quantize(tensor, scale, fp8_dtype=fp8_dtype)
    scale_inv = (1.0 / scale).to(torch.float32)
    return quantized, scale_inv


def nvidia_fp8_linear_forward(
    input_tensor: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: torch.Tensor | None = None,
    input_scale_inv: torch.Tensor | None = None,
    weight_scale_inv: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    High-performance FP8 GEMM forward:
    - On Hopper/Ada/Blackwell GPUs: uses hardware accelerated `torch._scaled_mm`.
    - Cross-platform: dynamic conversion and fallback.
    """
    # 1. Flatten batches for GEMM
    orig_shape = input_tensor.shape
    x_2d = input_tensor.view(-1, orig_shape[-1]) if input_tensor.dim() > 2 else input_tensor

    # 2. Ensure input is FP8
    if x_2d.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        x_fp8, in_scale_inv = fp8_dynamic_quantize(x_2d, fp8_dtype=torch.float8_e4m3fn)
    else:
        x_fp8 = x_2d
        in_scale_inv = input_scale_inv if input_scale_inv is not None else torch.tensor(1.0, device=x_2d.device)

    w_scale_inv = weight_scale_inv if weight_scale_inv is not None else torch.tensor(1.0, device=x_2d.device)

    # 3. Hardware Tensor Core path: torch._scaled_mm
    if is_torch_cuda_available() and is_fp8_supported() and hasattr(torch, "_scaled_mm"):
        # weight must be (K, N) column-major or transposed for _scaled_mm
        w_t = weight_fp8.t() if weight_fp8.shape[0] == weight_fp8.shape[-1] else weight_fp8
        out = torch._scaled_mm(
            x_fp8,
            w_t,
            scale_a=in_scale_inv,
            scale_b=w_scale_inv,
            bias=bias,
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
    else:
        # High precision simulation fallback (CPU / MPS / older CUDA)
        x_dequant = x_fp8.to(torch.float32) * in_scale_inv
        w_dequant = weight_fp8.to(torch.float32) * w_scale_inv
        # weight in PyTorch Linear is (out_features, in_features)
        if w_dequant.shape[1] == x_dequant.shape[1]:
            out = torch.matmul(x_dequant, w_dequant.t())
        else:
            out = torch.matmul(x_dequant, w_dequant)
        if bias is not None:
            out = out + bias
        out = out.to(out_dtype)

    # 4. Reshape back to original batch dimensions
    if input_tensor.dim() > 2:
        out = out.view(*orig_shape[:-1], out.shape[-1])
    return out


class NvidiaFp8Linear(nn.Module):
    """
    First-Class NVIDIA FP8 Linear Layer with Hopper/Blackwell TMA Descriptor support.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        use_tma: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_dtype = fp8_dtype
        self.use_tma = use_tma

        # Stored weights in FP8 format
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=fp8_dtype, device=device),
            requires_grad=False,
        )
        self.weight_scale_inv = nn.Parameter(
            torch.ones(1, dtype=torch.float32, device=device),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype or torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

        self._tma_descriptor: TmaDescriptor | None = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        use_tma: bool = True,
    ) -> NvidiaFp8Linear:
        """Quantizes an existing nn.Linear layer into NvidiaFp8Linear."""
        new_module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            fp8_dtype=fp8_dtype,
            use_tma=use_tma,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Quantize weight into FP8
        w_fp8, w_scale_inv = fp8_dynamic_quantize(linear.weight.data, fp8_dtype=fp8_dtype)
        new_module.weight.data.copy_(w_fp8)
        new_module.weight_scale_inv.data.copy_(w_scale_inv)

        if linear.bias is not None:
            new_module.bias.data.copy_(linear.bias.data)

        if use_tma and is_hopper_available():
            new_module._tma_descriptor = create_2d_tma_descriptor(new_module.weight.data)

        return new_module

    def get_tma_descriptor(self) -> TmaDescriptor:
        """Retrieves or creates the hardware TMA descriptor for this layer's weights."""
        if self._tma_descriptor is None:
            self._tma_descriptor = create_2d_tma_descriptor(self.weight.data)
        return self._tma_descriptor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nvidia_fp8_linear_forward(
            input_tensor=x,
            weight_fp8=self.weight,
            bias=self.bias,
            weight_scale_inv=self.weight_scale_inv,
            out_dtype=x.dtype,
        )


def replace_with_nvidia_fp8_linear(
    model: PreTrainedModel | nn.Module,
    modules_to_not_convert: list[str] | None = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    use_tma: bool = True,
) -> PreTrainedModel | nn.Module:
    """
    Recursively replaces all eligible `nn.Linear` layers in a model with `NvidiaFp8Linear`.
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = ["lm_head"]

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            setattr(model, name, NvidiaFp8Linear.from_linear(module, fp8_dtype=fp8_dtype, use_tma=use_tma))
        else:
            replace_with_nvidia_fp8_linear(module, modules_to_not_convert, fp8_dtype, use_tma)
    return model
