
file_path = '/Users/lodripeter/workspace/peterlodri-sec/transformers/src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py'
with open(file_path, 'r') as f:
    content = f.read()

new_content = content.replace('torch.split(tensor, lengths.tolist(), dim=2)', 'torch.split(tensor, kwargs.get("cu_seqlens_list", lengths.tolist()), dim=2)')

if new_content != content:
    with open(file_path, 'w') as f:
        f.write(new_content)
        print("Patched!")
else:
    print("Could not find lengths.tolist() in torch.split")
