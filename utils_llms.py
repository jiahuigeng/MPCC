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


class WhisperProcessor(ProcessorMixin):
    attributes = ["feature_extractor"]
    feature_extractor_class = "WhisperFeatureExtractor"
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        return self.tokenizer.get_decoder_prompt_ids(task=task, language=language, no_timestamps=no_timestamps)

    def get_T_after_cnn(self,L_in, dilation=1):
        for (padding, kernel_size, stride) in eval("[(1,3,1)] + [(1,3,2)] "):
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    def __call__(self, *args, **kwargs):
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        audio = kwargs.pop("audio", None)
        sampling_rate = kwargs.pop("sampling_rate", 16000)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            audio = args[0]
            args = args[1:]

        if audio is None and text is None:
            raise ValueError("You need to specify either an `audio` or `text` input to process.")

        if audio is not None:
            L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
            mel_len = L // 160
            audio_len_after_cnn = self.get_T_after_cnn(mel_len)
            audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
            inputs = self.feature_extractor(audio, *args, sampling_rate=sampling_rate, **kwargs)
            inputs['audio_len_after_cnn'] = torch.tensor(audio_len_after_cnn, dtype=torch.long)
            inputs['audio_token_num'] = torch.tensor(audio_token_num, dtype=torch.long)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs

        elif audio is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def get_prompt_ids(self, text: str, return_tensors="np"):
        return self.tokenizer.get_prompt_ids(text, return_tensors=return_tensors)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_audio(audio_file, audio_processor):
    audio_values, _ = librosa.load(audio_file, sr=16000) # sample rate should be 16000
    
    audio_process_values = audio_processor(audio_values, sampling_rate=16000, return_tensors="pt")
    input_features = audio_process_values['input_features']
    audio_len_after_cnn = audio_process_values['audio_len_after_cnn']
    audio_token_num = audio_process_values['audio_token_num']
                

    audio_input = {'audio_values': input_features, 
                   'audio_len_after_cnn': audio_len_after_cnn, 
                   'audio_token_num': audio_token_num, 
                   } 
    return audio_input


class InternOmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "InternOmni"):
        super().__init__(model_name, hf_path="OpenGVLab/InternOmni")
        self.image_size = 448
        self.max_num = 12

    def load_model(self):
        if self.model is None:
            print(f"Loading {self.model_name} from {self.hf_path}...")
            try:
                # Patch for missing transformers.onnx in newer transformers versions (e.g. 5.0.0)
                # OpenGVLab/InternOmni remote code depends on it
                import sys
                import types
                try:
                    import transformers.onnx
                except ImportError:
                    print("Patching missing transformers.onnx for InternOmni...")
                    dummy_onnx = types.ModuleType("transformers.onnx")
                    
                    class MockOnnxConfig:
                        def __init__(self, *args, **kwargs): pass
                    
                    class MockOnnxSeq2SeqConfigWithPast:
                        def __init__(self, *args, **kwargs): pass
                        
                    dummy_onnx.OnnxConfig = MockOnnxConfig
                    dummy_onnx.OnnxSeq2SeqConfigWithPast = MockOnnxSeq2SeqConfigWithPast
                    
                    sys.modules["transformers.onnx"] = dummy_onnx
                    import transformers
                # Suppress logging to avoid "KeyError: 'architectures'" in InternOmni config __repr__
                # which is triggered by logger.info(f"Model config {config}") in AutoConfig.from_pretrained
                # due to a bug in the remote code's config class default initialization
                transformers.logging.set_verbosity_error()

                import torch
                from transformers import AutoModel, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True, use_fast=False)
                # InternOmni usually requires float16/bfloat16
                self.model = AutoModel.from_pretrained(
                    self.hf_path, 
                    trust_remote_code=True, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                ).eval()
                
                # Audio processor
                self.audio_processor = WhisperProcessor.from_pretrained(self.hf_path)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading InternOmni: {e}")

    def generate(self, prompt: str, images: Optional[List[str]] = None, audio: Optional[str] = None) -> str:
        self.load_model()
        if not self.model:
            return "Error: Model not loaded."

        try:
            import torch
            
            # Prepare Image
            pixel_values = None
            if images and len(images) > 0:
                # Use load_image helper
                pixel_values = load_image(images[0], input_size=self.image_size, max_num=self.max_num)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Prepare Audio
            audio_input = None
            if audio:
                # Use load_audio helper
                audio_input = load_audio(audio, self.audio_processor)
                # Move tensors to cuda
                if isinstance(audio_input, dict):
                    if 'audio_values' in audio_input:
                        audio_input['audio_values'] = audio_input['audio_values'].to(torch.bfloat16).cuda()
                    if 'audio_len_after_cnn' in audio_input:
                        audio_input['audio_len_after_cnn'] = audio_input['audio_len_after_cnn'].to(self.model.device)
                    if 'audio_token_num' in audio_input:
                        audio_input['audio_token_num'] = audio_input['audio_token_num'].to(self.model.device)

            generation_config = dict(
                max_new_tokens=1024,
                do_sample=True,
                # temperature=0.7,
                # top_p=0.95,
            )
            
            # Construct input logic based on available modalities
            if hasattr(self.model, 'Audio_chat'):
                 # model.Audio_chat(tokenizer=tokenizer, pixel_values=pixel_values, audio=audio, question=None, generation_config)
                 # Note: User example passes question=None when audio is present (ASR task).
                 # But if we want to ask a question about image/audio, we should pass prompt.
                 # User example: question = '请将这段语音识别成文字...' then passed question=None in the line:
                 # response = model.Audio_chat(..., question=None, generation_config)
                 # Wait, user commented out the question line. Maybe for ASR it's None.
                 # But for general chat, we should pass the prompt.
                 # Let's pass the prompt.
                 
                 response = self.model.Audio_chat(
                     tokenizer=self.tokenizer, 
                     pixel_values=pixel_values,
                     audio=audio_input, 
                     question=prompt if prompt else None, 
                     generation_config=generation_config
                 )
                 return response
            else:
                 return "Error: model.Audio_chat method not found."

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error: {e}"

class MingOmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "Ming-Omni"):
        super().__init__(model_name, hf_path="inclusionAI/Ming-Lite-Omni") # Based on search

class M2OmniLLM(LocalHuggingFaceLLM):
    def __init__(self, model_name: str = "M2-omni"):
        super().__init__(model_name, hf_path="M2-omni/M2-omni") # Hypothetical path

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
        # "gpt-4o-mini",
        # "gemini-2.0-flash-exp",
        # "Qwen/Qwen2.5-Omni-7B",
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
