
from utils_llms import Qwen25OmniLLM
import traceback

def test_loading():
    print("Testing Qwen2.5-Omni-7B loading with 4-bit quantization...")
    try:
        model = Qwen25OmniLLM()
        # Just load, don't generate to save time if loading works
        model.load_model()
        if model.model is not None:
            print("Successfully loaded model!")
        else:
            print("Failed to load model (model is None).")
    except Exception as e:
        print(f"Error testing loading: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
