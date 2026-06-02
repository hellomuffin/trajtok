import argparse
from PIL import Image
import requests

import torch

from transformers import AutoProcessor, AutoModelForImageTextToText


video_path = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"
image1_path = "https://picsum.photos/id/237/536/354"
image2_path = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"


def main():
    parser = argparse.ArgumentParser(
        description="Test Molmo2 HF-compatible model."
    )
    parser.add_argument("checkpoint_dir", help="Location of Molmo2 checkpoint.")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.checkpoint_dir,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        padding_side="left",
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint_dir,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
    )

    single_image_messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text="Describe this image."),
                dict(type="image", image=Image.open(requests.get(image1_path, stream=True).raw)),
            ]
        }
    ]

    multi_image_messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text="Compare these images."),
                dict(
                    type="image",
                    image=Image.open(requests.get(image1_path, stream=True).raw),
                ),
                dict(
                    type="image",
                    image=Image.open(requests.get(image2_path, stream=True).raw),
                ),
            ],
        }
    ]

    video_messages = [
        {
            "role": "user",
            "content": [
                dict(type="text", text="Which animal appears in the video?"),
                dict(type="video", video=video_path),
            ]
        }
    ]

    single_image_inputs = processor.apply_chat_template(
        single_image_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    single_image_inputs = {k: v.to(model.device) for k, v in single_image_inputs.items()}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            single_image_generated_ids = model.generate(**single_image_inputs, max_new_tokens=448)
    single_image_generated_tokens = single_image_generated_ids[:, single_image_inputs['input_ids'].size(1):]
    single_image_generated_text = processor.post_process_image_text_to_text(
        single_image_generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(single_image_generated_text)

    multi_image_inputs = processor.apply_chat_template(
        multi_image_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    multi_image_inputs = {k: v.to(model.device) for k, v in multi_image_inputs.items()}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            multi_image_generated_ids = model.generate(**multi_image_inputs, max_new_tokens=448)
    multi_image_generated_tokens = multi_image_generated_ids[:, multi_image_inputs['input_ids'].size(1):]
    multi_image_generated_text = processor.post_process_image_text_to_text(
        multi_image_generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(multi_image_generated_text)

    video_inputs = processor.apply_chat_template(
        video_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    video_inputs = {k: v.to(model.device) for k, v in video_inputs.items()}
    
    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            video_generated_ids = model.generate(**video_inputs, max_new_tokens=2048)
    video_generated_tokens = video_generated_ids[:, video_inputs['input_ids'].size(1):]
    video_generated_text = processor.post_process_image_text_to_text(
        video_generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(video_generated_text)


if __name__ == "__main__":
    main()