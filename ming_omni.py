import torch
from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration


MODEL_ID = "inclusionAI/Ming-Lite-Omni"
DEVICE = "cuda"  # 没 GPU 就改成 "cpu"


def build():
    dtype = torch.bfloat16
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    return model, processor, dtype


@torch.inference_mode()
def run(messages, model, processor, dtype, use_whisper_encoder=False):
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True
    )
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(
        messages
    )

    audio_kwargs = {"use_whisper_encoder": True} if use_whisper_encoder else None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs=audio_kwargs,
    ).to(DEVICE)

    # bf16 对齐（和官方示例一致）
    for k in ("pixel_values", "pixel_values_videos", "audio_feats"):
        if k in inputs:
            inputs[k] = inputs[k].to(dtype)

    gen_cfg = GenerationConfig.from_dict(
        {"no_repeat_ngram_size": 10}
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        eos_token_id=processor.gen_terminator,
        generation_config=gen_cfg,
        use_whisper_encoder=use_whisper_encoder,
    )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    return processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def image_qa(model, processor, dtype):
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": "view.jpg"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        }
    ]
    out = run(messages, model, processor, dtype)
    print("🖼 Image QA result:\n", out)


def audio_qa(model, processor, dtype):
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Please summarize what is said in this audio."},
                {"type": "audio", "audio": "speechQA_sample.wav"},
            ],
        }
    ]
    out = run(messages, model, processor, dtype)
    print("🔊 Audio QA result:\n", out)


def main():
    model, processor, dtype = build()

    print("Running Image QA...")
    image_qa(model, processor, dtype)

    print("\nRunning Audio QA...")
    audio_qa(model, processor, dtype)


if __name__ == "__main__":
    main()
