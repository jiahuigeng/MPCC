import os
import base64
import json
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union

# Try to import api_sources for keys
try:
    import api_sources
except ImportError:
    api_sources = None

def get_key(name: str) -> Optional[str]:
    """Helper to get API key from api_sources.py or environment variable."""
    if api_sources and hasattr(api_sources, name):
        return getattr(api_sources, name)
    return os.environ.get(name)

# -----------------------------------------------------------------------------
# Base Class
# -----------------------------------------------------------------------------

class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        """
        Generate response from the model.
        
        Args:
            prompt (str): The text prompt.
            images (List[str], optional): List of image paths.
            audio (str, optional): Path to the audio file.
            
        Returns:
            str: The generated text response.
        """
        pass

# -----------------------------------------------------------------------------
# OpenAI Wrapper (GPT-4o-mini)
# -----------------------------------------------------------------------------

class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name)
        try:
            from openai import OpenAI
            api_key = get_key("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        if audio:
            print(f"WARNING: Model {self.model_name} does not natively support audio input via this wrapper. Audio ignored.")
            # Note: GPT-4o-audio-preview supports audio, but gpt-4o-mini is text/image.
        
        messages = []
        content = [{"type": "text", "text": prompt}]

        if images:
            for img_path in images:
                base64_image = self._encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        messages.append({"role": "user", "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

# -----------------------------------------------------------------------------
# Gemini Wrapper (Gemini 2.0)
# -----------------------------------------------------------------------------

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        super().__init__(model_name)
        try:
            import google.generativeai as genai
            api_key = get_key("GOOGLE_API_KEY")
            if not api_key:
                # Try to look for GEMINI_API_KEY as well
                api_key = get_key("GEMINI_API_KEY")
            
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found.")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")

    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        import google.generativeai as genai
        from PIL import Image

        parts = [prompt]

        if images:
            for img_path in images:
                try:
                    parts.append(Image.open(img_path))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        if audio:
            # For Gemini, we usually upload the file using the File API for large files,
            # or pass data if small. Here we assume we might need to upload.
            # Simplified approach: Upload to File API (requires internet)
            try:
                # Check if file exists
                if os.path.exists(audio):
                    # Note: In a real script, we should cache uploads or handle cleanup.
                    # Here we upload every time for simplicity, or check if it's already a File object.
                    print(f"Uploading audio {audio} to Gemini...")
                    audio_file = genai.upload_file(path=audio)
                    # Wait for processing if needed (video usually needs it, audio is fast)
                    parts.append(audio_file)
                else:
                    print(f"Audio file not found: {audio}")
            except Exception as e:
                print(f"Error handling audio: {e}")

        try:
            response = self.model.generate_content(parts)
            return response.text
        except Exception as e:
            return f"Error: {e}"

# -----------------------------------------------------------------------------
# Qwen-Omni Wrapper (DashScope API or Local)
# -----------------------------------------------------------------------------

class QwenOmniLLM(BaseLLM):
    def __init__(self, model_name: str = "qwen-omni-turbo"):
        # Assuming usage of Alibaba Cloud DashScope API
        super().__init__(model_name)
        try:
            import dashscope
            api_key = get_key("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY not found.")
            dashscope.api_key = api_key
        except ImportError:
            # If not installed, we can't use it.
            # Or we can assume local transformers usage if the user prefers.
            pass
            
    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        try:
            from dashscope import MultiModalConversation
            
            messages = [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
            
            # DashScope format for images/audio
            if images:
                for img in images:
                    # DashScope usually expects file:// or http://
                    # We assume local paths need file://
                    messages[0]["content"].append({"image": f"file://{os.path.abspath(img)}"})
            
            if audio:
                messages[0]["content"].append({"audio": f"file://{os.path.abspath(audio)}"})

            response = MultiModalConversation.call(
                model=self.model_name,
                messages=messages
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content[0]["text"]
            else:
                return f"Error: {response.code} - {response.message}"
        except ImportError:
            return "Error: dashscope library not installed. pip install dashscope"
        except Exception as e:
            return f"Error: {e}"

# -----------------------------------------------------------------------------
# Placeholder Wrappers for Research Models (Local Transformers)
# -----------------------------------------------------------------------------

class LocalHuggingFaceLLM(BaseLLM):
    """Base class for local transformers-based Omni models."""
    def __init__(self, model_name: str, hf_path: str):
        super().__init__(model_name)
        self.hf_path = hf_path
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        # Lazy loading
        if self.model is None:
            print(f"Loading {self.model_name} from {self.hf_path}...")
            try:
                from transformers import AutoModel, AutoTokenizer, AutoProcessor
                # This is generic; specific models might need specific classes
                self.model = AutoModel.from_pretrained(self.hf_path, trust_remote_code=True, device_map="auto")
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True)
                try:
                    self.processor = AutoProcessor.from_pretrained(self.hf_path, trust_remote_code=True)
                except:
                    pass
            except ImportError:
                print("Error: transformers/torch not installed.")
            except Exception as e:
                print(f"Error loading model: {e}")

    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        self.load_model()
        if not self.model:
            return "Error: Model not loaded."
        
        # Placeholder logic - generic inference
        # In reality, each Omni model (Ming, Intern, M2) has specific input formatting.
        return f"Response from {self.model_name} (Simulation): processed {prompt}, img={len(images or [])}, audio={bool(audio)}"


class InternOmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "InternOmni"):
        # Assuming generic path or user provides it
        super().__init__(model_name, hf_path="OpenGVLab/InternOmni") # Hypothetical path

class MingOmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "Ming-Omni"):
        super().__init__(model_name, hf_path="inclusionAI/Ming-Lite-Omni") # Based on search

class M2OmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "M2-omni"):
        super().__init__(model_name, hf_path="M2-omni/M2-omni") # Hypothetical path

class Qwen25OmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B"):
        super().__init__(model_name, hf_path="Qwen/Qwen2.5-Omni-7B")

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def get_model(model_name: str) -> BaseLLM:
    name_lower = model_name.lower()
    
    if "gpt" in name_lower or "openai" in name_lower:
        return OpenAILLM(model_name)
    
    if "gemini" in name_lower:
        return GeminiLLM(model_name)
    
    if "qwen" in name_lower and "omni" in name_lower:
        if "2.5" in name_lower or "huggingface" in name_lower:
             return Qwen25OmniLLM(model_name)
        return QwenOmniLLM(model_name)
    
    if "intern" in name_lower and "omni" in name_lower:
        return InternOmniLLM(model_name)
        
    if "ming" in name_lower:
        return MingOmniLLM(model_name)
        
    if "m2" in name_lower:
        return M2OmniLLM(model_name)

    # Default fallback
    print(f"Warning: Unknown model {model_name}, defaulting to OpenAI.")
    return OpenAILLM("gpt-4o-mini")

# -----------------------------------------------------------------------------
# Test Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Test all models
    test_models = [
        "gpt-4o-mini",
        "gemini-2.0-flash-exp",
        "qwen-omni-turbo",
        "Qwen/Qwen2.5-Omni-7B",
        "InternOmni",
        "Ming-Omni",
        "M2-omni"
    ]
    
    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    prompt_text = "Please describe the image."
    
    # Download image temporarily for testing
    import requests
    from PIL import Image
    from io import BytesIO
    
    local_image_path = "test_view.jpg"
    print(f"Downloading test image from {image_url}...")
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(local_image_path, "wb") as f:
                f.write(response.content)
            print(f"Saved to {local_image_path}")
        else:
            print(f"Failed to download image: {response.status_code}")
            local_image_path = None
    except Exception as e:
        print(f"Error downloading image: {e}")
        local_image_path = None

    if local_image_path:
        for model_name in test_models:
            print(f"\n--- Testing Model: {model_name} ---")
            try:
                model = get_model(model_name)
                # Pass list of images as expected by signature
                response = model.generate(prompt=prompt_text, images=[local_image_path])
                print(f"Response:\n{response}")
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
        
        # Cleanup
        # if os.path.exists(local_image_path):
        #    os.remove(local_image_path)
    else:
        print("Skipping tests because image could not be downloaded.")
