import re

file_path = '/Users/lodripeter/workspace/peterlodri-sec/transformers/src/transformers/vision_utils.py'
with open(file_path, 'r') as f:
    content = f.read()

content = content.replace(
    'cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)\n    max_seqlen = get_max_seqlen(cu_seqlens, config, kwargs=kwargs)\n    return cu_seqlens, max_seqlen',
    'cu_seqlens = get_vision_cu_seqlens(grid_thw, merge_temporal=merge_temporal, kwargs=kwargs)\n    max_seqlen = get_max_seqlen(cu_seqlens, config, kwargs=kwargs)\n    if kwargs is not None and "cu_seqlens_list" not in kwargs:\n        kwargs["cu_seqlens_list"] = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()\n    return cu_seqlens, max_seqlen'
)

with open(file_path, 'w') as f:
    f.write(content)
