import inspect
from transformers import AutoProcessor

def inspect_sam3_session_api():
    model_path = "./checkpoints/sam3"
    print(f"Loading Processor from {model_path}...")

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # List of methods we suspect are needed for the workflow
        methods_to_inspect = [
            "init_video_session",
            "add_text_prompt",
            "postprocess_outputs"
        ]

        print("\n=== PROCESSOR API SIGNATURES ===")
        for method_name in methods_to_inspect:
            if hasattr(processor, method_name):
                method = getattr(processor, method_name)
                sig = inspect.signature(method)
                print(f"\n[ {method_name} ]")
                for name, param in sig.parameters.items():
                    print(f"  - {name}: {param.annotation}")
            else:
                print(f"\n[X] Method '{method_name}' not found.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_sam3_session_api()