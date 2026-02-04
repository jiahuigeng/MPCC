import sys
import types

def test_patch():
    # Simulate missing transformers.onnx
    if "transformers.onnx" in sys.modules:
        del sys.modules["transformers.onnx"]
        
    try:
        import transformers.onnx
        print("transformers.onnx exists (unexpected if we just deleted it, but maybe it reloaded)")
    except ImportError:
        print("transformers.onnx missing as expected")

    # Apply patch
    print("Applying patch...")
    dummy_onnx = types.ModuleType("transformers.onnx")
    
    class MockOnnxConfig:
        pass
    class MockOnnxSeq2SeqConfigWithPast:
        pass
        
    dummy_onnx.OnnxConfig = MockOnnxConfig
    dummy_onnx.OnnxSeq2SeqConfigWithPast = MockOnnxSeq2SeqConfigWithPast
    
    sys.modules["transformers.onnx"] = dummy_onnx
    
    # Verify import
    try:
        from transformers.onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
        print("Successfully imported OnnxConfig and OnnxSeq2SeqConfigWithPast from patched module")
    except ImportError as e:
        print(f"Failed to import after patch: {e}")

if __name__ == "__main__":
    test_patch()
