from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.mesh import MeshRouterWrapper


def main():
    print("🕸️ Instantiating Mesh Router Wrapper Example 🕸️\n")

    # We will instantiate two dummy TinyLlama or similar small models for the example.
    # To keep this fast, we will just use randomly initialized experts.
    model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

    print(f"Loading Base Configuration for {model_id}...")
    try:
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        print(f"Failed to load {model_id}. Ensure you are authenticated or have internet access.")
        return

    # Increase vocab size slightly if tokenizer is padded, but tiny-random usually works
    config.n_experts = 2  # Our mesh router expects this property

    print("Initializing Expert 1...")
    expert_1 = AutoModelForCausalLM.from_config(config)
    print("Initializing Expert 2...")
    expert_2 = AutoModelForCausalLM.from_config(config)

    experts = [expert_1, expert_2]

    print("\nWrapping experts in MeshRouterWrapper...")
    mesh_model = MeshRouterWrapper(config, experts)

    prompt = "The quick brown fox"
    inputs = tokenizer(prompt, return_tensors="pt")

    print(f"\nGenerating text with Mesh Router from prompt: '{prompt}'")
    # Generate using the unified MeshRouterWrapper which delegates down to experts
    outputs = mesh_model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.7)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGeneration successful!\nResult: {response}")


if __name__ == "__main__":
    main()
