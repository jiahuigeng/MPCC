import os
import base64
import json
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import librosa
from transformers.processing_utils import ProcessorMixin

# Try to import api_sources or api_resources for keys
try:
    import api_sources
except ImportError:
    api_sources = None

try:
    import api_resources
except ImportError:
    api_resources = None

def get_key(name: str) -> Optional[Union[str, List[str]]]:
    """Helper to get API key from api_sources/api_resources or environment variable."""
    # Check api_resources first (user preference)
    if api_resources and hasattr(api_resources, name):
        return getattr(api_resources, name)
    # Check api_sources
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
    def generate(self, prompt: str, images: Optional[List[Union[str, bytes]]] = None, audio: Optional[str] = None, flag: str = "text") -> str:
        """
        Generate response from the model.
        
        Args:
            prompt (str): The text prompt.
            images (List[Union[str, bytes]], optional): List of image paths or image bytes.
            audio (str, optional): Path to the audio file.
            flag (str, optional): Mode flag. 
                - "text": Text + Image (standard VQA).
                - "spoken": Audio + Image (Multimodal interaction).
                - "asr": ASR -> Text -> LLM (Pipeline: Audio to Text, then Text+Image to LLM).
            
        Returns:
            str: The generated text response.
        """
        pass

# -----------------------------------------------------------------------------
# OpenAI Wrapper (GPT-4o-mini / GPT-4o)
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

    def _encode_image(self, image_input: Union[str, bytes]) -> str:
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def generate(self, prompt: str, images: Optional[List[Union[str, bytes]]] = None, audio: Optional[str] = None, flag: str = "text") -> str:
        # 1. Handle "asr" flag: Explicitly pipeline Audio -> Text -> Vision
        if flag == "asr" and audio:
            print(f"Info: OpenAI [ASR Mode] - Transcribing audio first...")
            try:
                # Step 1: Transcribe
                audio_instruction = self._process_audio_instruction(audio)
                print(f"  -> Transcribed Audio: '{audio_instruction}'")
                
                # Step 2: Update prompt
                prompt = f"{prompt}\n\n[Audio Transcript]: {audio_instruction}"
                
                # Step 3: Clear audio so we don't send it to the model again
                audio = None
                
                # Continue to standard processing (Text + Image)
                # Ensure we use a vision-capable model if the current one is audio-only
                if "audio" in self.model_name:
                    # Temporary switch for the vision part if the user selected an audio model
                    # But ideally, 'asr' mode implies using a text/vision model for the second step.
                    pass
            except Exception as e:
                return f"Error in ASR step: {e}"

        # 2. Handle "spoken" flag: Attempt native Audio support
        # Note: OpenAI does not support Audio + Image in one call (REST API).
        if flag == "spoken":
            if images and audio:
                print("Warning: OpenAI does not support native Audio + Image in one request.")
                # We will proceed to try sending both, but it will likely fail or require the model to be audio-only (no images)
                # or vision-only (no audio).
                # However, to be helpful, if the user insists on 'spoken' with OpenAI, we might fallback to the pipeline
                # OR we just let the API error out/warn to show capability limits.
                # Given previous context, let's allow it to proceed so the user sees the result (or lack thereof).
                pass
        
        messages = []
        content = []
        if prompt:
            content.append({"type": "text", "text": prompt})

        # Handle Images
        if images:
            # Check for OpenAI models that don't support images
            # If we are in 'spoken' mode with an audio model, this might fail.
            # If we are in 'asr' mode, we already cleared audio, so we should be fine assuming we pick a vision model.
            
            override_model = None
            if "audio-preview" in self.model_name:
                 if flag == "asr" or flag == "text":
                     # If we are doing ASR (converted to text) or Text mode, we MUST use a vision model
                     print("  -> Switching to 'gpt-4o' for Image processing (Audio model doesn't support images)...")
                     override_model = "gpt-4o"
                 elif flag == "spoken":
                     # In spoken mode, the user *wants* to test the audio model.
                     # But audio-preview doesn't support images. 
                     # We will warn and skip images? Or let it fail?
                     # Let's warn.
                     print("  -> Warning: 'gpt-4o-audio-preview' does not support input images. Images might be ignored or cause error.")
            
            for img in images:
                base64_image = self._encode_image(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        else:
            override_model = None
        
        # Handle Audio (Native)
        if audio and flag == "spoken":
            # Only send audio payload if flag is 'spoken' (or if we didn't clear it in ASR mode)
            try:
                with open(audio, "rb") as f:
                    audio_bytes = f.read()
                base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
                content.append({
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64_audio,
                        "format": "wav"
                    }
                })
            except Exception as e:
                print(f"Error processing audio for OpenAI: {e}")
        
        messages.append({"role": "user", "content": content})

        try:
            # Use override_model if set, otherwise self.model_name
            model_to_use = override_model if override_model else self.model_name
            
            # Final check
            if audio and "gpt-4o" in model_to_use and "audio" not in model_to_use:
                 if flag == "spoken":
                     return "Error: 'gpt-4o' (Text/Vision) does not support Audio input. Use 'gpt-4o-audio-preview' or flag='asr'."
            
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=4096
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

    def _process_audio_instruction(self, audio_path: str) -> str:
        """Helper to transcribe/extract instruction from audio using OpenAI Whisper (whisper-1)."""
        try:
            with open(audio_path, "rb") as audio_file:
                # Use Whisper-1 for accurate ASR
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            return transcription.text
        except Exception as e:
            raise RuntimeError(f"Failed to process audio instruction: {e}")

# -----------------------------------------------------------------------------
# Gemini Wrapper (Gemini 2.0)
# -----------------------------------------------------------------------------

class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        super().__init__(model_name)
        self.use_new_sdk = False
        self.client = None
        self.model = None

        # API Key
        self.api_key = get_key("GOOGLE_API_KEY") or get_key("GEMINI_API_KEY") or get_key("GEMINI_API_KEYS")
        
        # Handle list of keys (random selection)
        if isinstance(self.api_key, list):
            import random
            self.api_key = random.choice(self.api_key)
            
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY(S) not found.")

        # Try new SDK first: google-genai
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            self.use_new_sdk = True
            print("Using google-genai SDK (v2).")
        except ImportError:
            # Fallback to old SDK: google-generativeai
            try:
                import google.generativeai as genai_old
                genai_old.configure(api_key=self.api_key)
                self.model = genai_old.GenerativeModel(model_name)
                print("Using google-generativeai SDK (legacy).")
            except ImportError:
                raise ImportError("Please install google-genai or google-generativeai.")

    def generate(self, prompt: str, images: Optional[List[Union[str, bytes]]] = None, audio: Optional[str] = None, flag: str = "text") -> str:
        parts = []
        
        # 1. Handle "asr" flag for Gemini: Simulate Pipeline (Transcribe -> Text+Image)
        if flag == "asr" and audio:
            print("Info: Gemini [ASR Mode] - Transcribing audio first...")
            try:
                # Transcribe using Gemini itself (audio only)
                transcription = self.generate(prompt="Please transcribe this audio exactly.", audio=audio, flag="spoken") 
                # Note: recursive call with 'spoken' and no images to get transcript
                print(f"  -> Transcribed Audio: '{transcription}'")
                
                prompt = f"{prompt}\n\n[Audio Transcript]: {transcription}"
                audio = None # Clear audio for the main call
            except Exception as e:
                return f"Error in Gemini ASR step: {e}"
        
        parts.append(prompt)
        
        # Load Images
        if images:
            from PIL import Image
            import io
            for img_input in images:
                try:
                    if isinstance(img_input, str):
                        parts.append(Image.open(img_input))
                    elif isinstance(img_input, bytes):
                        parts.append(Image.open(io.BytesIO(img_input)))
                    elif isinstance(img_input, Image.Image):
                        parts.append(img_input)
                    else:
                        print(f"Warning: Unsupported image input type {type(img_input)}")
                except Exception as e:
                    print(f"Error loading image {img_input if isinstance(img_input, str) else 'bytes'}: {e}")

        # New SDK Implementation
        if self.use_new_sdk:
            from google.genai import types
            
            # Handle Audio for new SDK
            if audio and flag == "spoken":
                try:
                    with open(audio, "rb") as f:
                        audio_bytes = f.read()
                    parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))
                except Exception as e:
                     print(f"Error loading audio {audio} for Gemini New SDK: {e}")

            try:
                # The new SDK expects 'contents'
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts
                )
                return response.text
            except Exception as e:
                return f"Error (new SDK): {e}"

        # Old SDK Implementation
        else:
            import google.generativeai as genai_old
            if audio and flag == "spoken":
                try:
                    if os.path.exists(audio):
                        print(f"Uploading audio {audio} to Gemini (Legacy)...")
                        audio_file = genai_old.upload_file(path=audio)
                        parts.append(audio_file)
                    else:
                        print(f"Audio file not found: {audio}")
                except Exception as e:
                    print(f"Error handling audio: {e}")

            try:
                response = self.model.generate_content(parts)
                return response.text
            except Exception as e:
                return f"Error (old SDK): {e}"

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
    def __init__(self, model_name: str, hf_path: str, model_class_name: str = "AutoModel"):
        super().__init__(model_name)
        self.hf_path = hf_path
        self.model_class_name = model_class_name
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        # Lazy loading
        if self.model is None:
            print(f"Loading {self.model_name} from {self.hf_path}...")
            try:
                from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoProcessor
                
                loader = AutoModel
                if self.model_class_name == "AutoModelForCausalLM":
                    loader = AutoModelForCausalLM
                
                # This is generic; specific models might need specific classes
                self.model = loader.from_pretrained(self.hf_path, trust_remote_code=True, device_map="auto")
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







class Qwen25OmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B"):
        super().__init__(model_name, hf_path="Qwen/Qwen2.5-Omni-7B", model_class_name="Qwen2_5OmniForConditionalGeneration")
        self.use_audio_in_video = True # Default setting from example

    def load_model(self):
        # Lazy loading
        if self.model is None:
            print(f"Loading {self.model_name} from {self.hf_path}...")
            try:
                from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
                import torch

                # Patch for missing set_submodule in Qwen2_5OmniForConditionalGeneration
                # This is required for bitsandbytes quantization on some custom models or torch versions
                if not hasattr(Qwen2_5OmniForConditionalGeneration, 'set_submodule'):
                    print("Patching set_submodule for Qwen2_5OmniForConditionalGeneration")
                    def set_submodule(self, target, module):
                        atoms = target.split(".")
                        name = atoms.pop()
                        mod = self
                        for item in atoms:
                            mod = getattr(mod, item)
                        setattr(mod, name, module)
                    Qwen2_5OmniForConditionalGeneration.set_submodule = set_submodule

                print(f"Loading {self.model_name} with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )

                # Load config first to patch potential issues
                config = AutoConfig.from_pretrained(self.hf_path, trust_remote_code=True)
                
                # Patch for 'pad_token_id' missing in TalkerConfig
                if hasattr(config, 'talker_config') and config.talker_config is not None:
                    # Check vocab_size of talker_config
                    talker_vocab_size = getattr(config.talker_config, 'vocab_size', 4096)
                    
                    if not hasattr(config.talker_config, 'pad_token_id') or config.talker_config.pad_token_id is None:
                        # Use a safe default (e.g., 0 or size-1) instead of main model's 151643
                        # Because Talker has much smaller vocab size (e.g. 4096 or 8192)
                        pad_token = 0 
                        if talker_vocab_size > 0:
                             pad_token = talker_vocab_size - 1
                             
                        setattr(config.talker_config, 'pad_token_id', pad_token)
                        print(f"Patched talker_config with pad_token_id={pad_token} (vocab_size={talker_vocab_size})")
                    else:
                        # Even if it exists, check if it's valid
                         current_pad = config.talker_config.pad_token_id
                         if current_pad >= talker_vocab_size:
                             print(f"Fixing invalid pad_token_id {current_pad} for talker vocab {talker_vocab_size}")
                             config.talker_config.pad_token_id = talker_vocab_size - 1

                # Load model
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    self.hf_path, 
                    config=config,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    use_safetensors=True  # Force safetensors to bypass torch.load vulnerability check
                )
                
                # Load processor
                self.processor = Qwen2_5OmniProcessor.from_pretrained(self.hf_path, trust_remote_code=True)
                
            except ImportError:
                print("Error: transformers, torch, bitsandbytes or qwen_omni_utils not installed.")
                print("Please install: pip install transformers qwen-omni-utils bitsandbytes accelerate")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading model: {e}")

    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        self.load_model()
        if not self.model:
            return "Error: Model not loaded."

        try:
            from qwen_omni_utils import process_mm_info
            
            # Construct conversation
            content = []
            
            # Handle images/videos
            # Note: The example treated video input specifically. 
            # We will treat 'images' list as images unless they look like video files, 
            # but for MPCC task they are likely images.
            if images:
                for img in images:
                    # Simple heuristic: if extension is mp4/avi/mov, treat as video
                    if img.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        content.append({"type": "video", "video": img})
                    else:
                        content.append({"type": "image", "image": img})
            
            # Handle audio
            if audio:
                 # Qwen2.5-Omni supports audio input? Example shows output audio generation.
                 # But let's assume standard Qwen multimodal format if supported.
                 # The user example shows video input. 
                 # If audio input is supported, it usually follows "audio" type.
                 # However, qwen-omni-utils process_mm_info handles 'audio', 'image', 'video'.
                 content.append({"type": "audio", "audio": audio})
            
            content.append({"type": "text", "text": prompt})
            
            conversation = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            # Preparation for inference
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # Process multimodal info
            # qwen_omni_utils.process_mm_info is key here
            audios, images_processed, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)
            
            # Create inputs
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images_processed, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=self.use_audio_in_video
            )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Inference: Generation of the output text and audio
            # We prioritize text output for this task
            output = self.model.generate(**inputs, use_audio_in_video=self.use_audio_in_video)
            
            # The model returns (text_ids, audio_values) tuple if generating both?
            # The example shows: text_ids, audio = model.generate(...)
            # Let's handle the return value carefully.
            
            if isinstance(output, tuple):
                 text_ids = output[0]
                 # audio_out = output[1] 
            else:
                 text_ids = output
            
            response_text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            return response_text

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error during generation: {e}"

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
    


    # Default fallback
    print(f"Warning: Unknown model {model_name}, defaulting to OpenAI.")
    return OpenAILLM("gpt-4o-mini")

# -----------------------------------------------------------------------------
# Test Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Test cases
    prompt_text = "Describe what you see in the image."
    prompt_audio = "output.wav" 
    prompt_images =["test_view.jpg"] 
    
    # Check if files exist
    if not os.path.exists(prompt_audio):
        print(f"Creating dummy audio file {prompt_audio}...")
        # Create a dummy wav file if it doesn't exist
        import wave
        with wave.open(prompt_audio, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b'\x00' * 32000) # 2 seconds of silence
            
    if not os.path.exists(prompt_images[0]):
        print(f"Creating dummy image file {prompt_images[0]}...")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(prompt_images[0])

    print(f"Testing with Audio: {prompt_audio} + Image: {prompt_images[0]}")
    print("-" * 50)

    # Define test configurations: (Name, Model, Flag, UseImages)
    models_to_test = [
        # ("OpenAI (Text+Image)", OpenAILLM("gpt-4o"), "text", True),
        # ("OpenAI (ASR Pipeline)", OpenAILLM("gpt-4o"), "asr", True),
        
        ("OpenAI (Text+Image)", OpenAILLM("gpt-4o-mini"), "text", True),
        ("OpenAI (ASR Pipeline)", OpenAILLM("gpt-4o-mini"), "asr", True),
        
        # Gemini 2.0 Flash
        # ("Gemini 2.0 Flash (Text+Image)", GeminiLLM("gemini-2.0-flash"), "text", True),
        # ("Gemini 2.0 Flash (Spoken - Image+Audio)", GeminiLLM("gemini-2.0-flash"), "spoken", True),
        # ("Gemini 2.0 Flash (ASR Pipeline)", GeminiLLM("gemini-2.0-flash"), "asr", True),

        # Gemini 2.5 Flash (Requested)
        ("Gemini 2.5 Flash (Image+Audio)", GeminiLLM("gemini-2.5-flash"), "spoken", True),
        ("Gemini 2.5 Flash (Text+Audio)",   GeminiLLM("gemini-2.5-flash"), "spoken", False), # No image
        ("Gemini 2.5 Flash (ASR+Audio)",    GeminiLLM("gemini-2.5-flash"), "asr", True),    # With image (VQA via ASR)

        # Gemini 2.5 Pro (Requested)
        ("Gemini 2.5 Pro (Image+Audio)", GeminiLLM("gemini-2.5-pro"), "spoken", True),
        ("Gemini 2.5 Pro (Text+Audio)",   GeminiLLM("gemini-2.5-pro"), "spoken", False), # No image
        ("Gemini 2.5 Pro (ASR+Audio)",    GeminiLLM("gemini-2.5-pro"), "asr", True),    # With image (VQA via ASR)
    ]

    for name, model, flag, use_images in models_to_test:
        print(f"\n--- Testing: {name} [Flag={flag}] ---")
        try:
            current_images = prompt_images if use_images else None
            response = model.generate(prompt=prompt_text, images=current_images, audio=prompt_audio, flag=flag)
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Failed: {e}")
