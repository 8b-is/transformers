import glob
import re

files = glob.glob('/Users/lodripeter/workspace/peterlodri-sec/transformers/src/transformers/models/**/*.py', recursive=True)

for file_path in files:
    with open(file_path, 'r') as f:
        content = f.read()

    if 'lengths.tolist()' in content:
        # We find block:
        # lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # splits = [
        #     torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        # ]
        
        # We just want to replace torch.split(tensor, lengths.tolist(), dim=2) 
        # But we need to define cu_seqlens_list first. 
        # The exact string is usually:
        # lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        # splits = [
        #     torch.split(tensor, lengths.tolist(), dim=2) ...
        # ]
        
        # Let's replace lengths.tolist() inside the torch.split with cu_seqlens_list.
        # But where to put the definition of cu_seqlens_list?
        # Just replace lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        
        # Using a regex
        new_content = re.sub(
            r'(lengths = [^\n]+\n\s*)(splits = \[\n\s*torch\.split\(tensor, )lengths\.tolist\(\)(, dim=\d+\) for tensor in \([^)]+\)\n\s*\])',
            r'\1cu_seqlens_list = kwargs.get("cu_seqlens_list")\n            if cu_seqlens_list is None:\n                cu_seqlens_list = lengths.tolist()\n            \2cu_seqlens_list\3',
            content
        )
        # Some might not have exactly that formatting, so let's do a more robust one
        if new_content != content:
            with open(file_path, 'w') as f:
                f.write(new_content)
                print(f"Patched {file_path}")
        else:
            print(f"Failed to patch {file_path} even though it contains lengths.tolist()")

