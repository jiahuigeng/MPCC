import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)

# ======================
# 基本配置
# ======================
MODEL_ID = "inclusionAI/Ming-Lite-Omni"
DEVICE = "cuda"  # 没 GPU 就改成 "cpu"
DTYPE = torch.bfloat16  # GPU 不支持 bf16 可改成 torch.float16

IMAGE_PATH = "view.jpg"
AUDIO_PATH = "speechQA_sample.wav"


# ======================
# 加载模型与处理器
# ======================
def load_model_and_processor():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        trust_remote_code=True,   # ⭐ 关键：自动加载 modeling_bailingmm.py
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    return model, processor


# ======================
# 通用推理函数
# ======================
@torch.inference_mode()
def run(messages, model, processor, use_whisper_encoder=False):
    # 构造 prompt
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    # 自动解析 image / audio
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

    # 对齐 dtype（官方示例就是这么做的）
    for k in ("pixel_values", "pixel_values_videos", "audio_feats"):
        if k in inputs:
            inputs[k] = inputs[k].to(DTYPE)

    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        no_repeat_ngram_size=10,
    )

    outputs = model.generate(
        **inputs,
        generation_config=gen_cfg,
        eos_token_id=processor.gen_terminator,
        use_whisper_encoder=use_whisper_encoder,
    )

    # 只解码新生成的部分
    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
    return processor.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# ======================
# Image QA
# ======================
def image_qa(model, processor):
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        }
    ]

    result = run(messages, model, processor)
    print("🖼 Image QA Result:")
    print(result)


# ======================
# Audio QA
# ======================
def audio_qa(model, processor):
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Please summarize what is said in this audio."},
                {"type": "audio", "audio": AUDIO_PATH},
            ],
        }
    ]

    result = run(messages, model, processor)
    print("🔊 Audio QA Result:")
    print(result)


# ======================
# ASR（可选）
# ======================
def asr(model, processor):
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": AUDIO_PATH},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }
    ]

    result = run(
        messages,
        model,
        processor,
        use_whisper_encoder=True,  # ⭐ ASR 必须开
    )
    print("📝 ASR Result:")
    print(result)


# ======================
# 主入口
# ======================
def main():
    model, processor = load_model_and_processor()

    print("\n===== Running Image QA =====")
    image_qa(model, processor)

    print("\n===== Running Audio QA =====")
    audio_qa(model, processor)

    # 如需 ASR，取消下面注释
    # print("\n===== Running ASR =====")
    # asr(model, processor)


if __name__ == "__main__":
    main()
