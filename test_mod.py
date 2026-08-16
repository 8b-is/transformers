import re

file_path = '/Users/lodripeter/workspace/peterlodri-sec/transformers/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py'
with open(file_path, 'r') as f:
    content = f.read()

# Modify Qwen3VLVisionAttention
content = re.sub(
    r'def forward\(\s*self,\s*hidden_states: torch\.Tensor,\s*cu_seqlens: torch\.Tensor,\s*position_embeddings: tuple\[torch\.Tensor, torch\.Tensor\] \| None = None,\s*max_seqlen: int \| None = None,\s*\*\*kwargs,\s*\) -> torch\.Tensor:',
    r'def forward(\n        self,\n        hidden_states: torch.Tensor,\n        cu_seqlens: torch.Tensor,\n        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,\n        max_seqlen: int | None = None,\n        cu_seqlens_list: list[int] | None = None,\n        **kwargs,\n    ) -> torch.Tensor:',
    content
)

# Modify Qwen3VLVisionAttention logic
content = re.sub(
    r'lengths = cu_seqlens\[1:\] - cu_seqlens\[:-1\]\n\s*splits = \[\n\s*torch\.split\(tensor, lengths\.tolist\(\), dim=2\) for tensor in \(query_states, key_states, value_states\)\n\s*\]',
    r'if cu_seqlens_list is None:\n                cu_seqlens_list = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()\n            splits = [\n                torch.split(tensor, cu_seqlens_list, dim=2) for tensor in (query_states, key_states, value_states)\n            ]',
    content
)

# Modify Qwen3VLVisionBlock
content = re.sub(
    r'def forward\(\s*self,\s*hidden_states: torch\.Tensor,\s*cu_seqlens: torch\.Tensor,\s*position_embeddings: tuple\[torch\.Tensor, torch\.Tensor\] \| None = None,\s*\*\*kwargs,\s*\) -> torch\.Tensor:',
    r'def forward(\n        self,\n        hidden_states: torch.Tensor,\n        cu_seqlens: torch.Tensor,\n        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,\n        **kwargs,\n    ) -> torch.Tensor:',
    content
)
# wait, block passes **kwargs to self.attn, so cu_seqlens_list will automatically be forwarded if we pass it to block.

# Modify Qwen3VisionModel
content = re.sub(
    r'cu_seqlens, max_seqlen = get_vision_attention_seqlens\(grid_thw, self\.config, kwargs=kwargs\)',
    r'cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)\n        cu_seqlens_list = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()',
    content
)

content = re.sub(
    r'hidden_states = blk\(\n\s*hidden_states,\n\s*cu_seqlens=cu_seqlens,\n\s*max_seqlen=max_seqlen,\n\s*position_embeddings=position_embeddings,\n\s*\*\*kwargs,\n\s*\)',
    r'hidden_states = blk(\n                hidden_states,\n                cu_seqlens=cu_seqlens,\n                max_seqlen=max_seqlen,\n                position_embeddings=position_embeddings,\n                cu_seqlens_list=cu_seqlens_list,\n                **kwargs,\n            )',
    content
)

with open('test_mod_out.py', 'w') as f:
    f.write(content)

