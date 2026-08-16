import torch
import sys
import os

# Add transformers to path
sys.path.insert(0, os.path.abspath("src"))

from transformers.integrations.bitnet import pack_weights, unpack_weights, VALUES_PER_ITEM

def test_packing_unpacking(size):
    print(f"Testing size {size}...")
    torch.manual_seed(0)
    # Generate random ternary weights {-1, 0, 1}
    w = torch.randint(-1, 2, (size, 256), dtype=torch.int8)
    
    w_packed = pack_weights(w)
    expected_rows = (size + VALUES_PER_ITEM - 1) // VALUES_PER_ITEM
    assert w_packed.shape == (expected_rows, 256), f"Expected {(expected_rows, 256)}, got {w_packed.shape}"
    
    w_unpacked = unpack_weights(w_packed, dtype=torch.int8)
    
    # slice to original size
    w_unpacked_sliced = w_unpacked[:size]
    
    if not torch.equal(w, w_unpacked_sliced):
        print(f"FAILED for size {size}")
        print("Original:\n", w[:2, :5])
        print("Unpacked:\n", w_unpacked_sliced[:2, :5])
        sys.exit(1)
    else:
        print(f"PASSED for size {size}")

test_packing_unpacking(1024)
test_packing_unpacking(1025)
test_packing_unpacking(1027)
test_packing_unpacking(7)
print("All tests passed.")
