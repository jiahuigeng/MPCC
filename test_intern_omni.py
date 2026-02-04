import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import librosa
from transformers.processing_utils import ProcessorMixin
import torch

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

path = 'OpenGVLab/InternOmni'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
audio_processor = WhisperProcessor.from_pretrained(path)
# set the max number of tiles in `max_num`
pixel_values = load_image('./1.jpg', max_num=12).to(torch.bfloat16).cuda()
audio = load_audio('./1.wav', audio_processor)
generation_config = dict(max_new_tokens=1024, do_sample=True)

# question = '请将这段语音识别成文字，并以文字形式展示出来。'
response = model.Audio_chat(tokenizer=tokenizer, pixel_values=pixel_values,audio=audio, question=None, generation_config=generation_config)
print(f'Assistant: {response}')
