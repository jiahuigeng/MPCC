
try:
    from utils_llms import InternOmniLLM, WhisperProcessor, load_image, load_audio
    print("Successfully imported InternOmniLLM and helpers")
    
    model = InternOmniLLM()
    print(f"Initialized InternOmniLLM with path: {model.hf_path}")
    
    # Check if helpers are callable
    import inspect
    print(f"load_image is function: {inspect.isfunction(load_image)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
