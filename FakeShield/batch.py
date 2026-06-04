import os
import json
import argparse
import torch
from tqdm import tqdm
from pathlib import Path
# импортируем функции из твоего файла
from llava.serve.cli import DTE_FDM_init, load_image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from transformers import TextStreamer


def detect_single_image(args, tokenizer, model, image_processor, DTG, model_name, image_path, output_path):
    conv = conv_templates[args.conv_mode].copy()

    image = load_image(image_path)
    label = DTG.predict(image_path)

    image_size = image.size
    image_tensor = process_images([image], image_processor, model.config)

    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    # тот же prompt как у них
    inp = "Was this photo taken directly from the camera without any processing? Has it been tampered with by any artificial photo modification techniques such as ps?"

    if label == 0:
        inp = "This is a picture that is suspected to have been tampered with by AIGC inpainting. " + inp
    elif label == 1:
        inp = "This is a picture that is suspected to have been tampered with by DeepFake. " + inp
    elif label == 2:
        inp = "This is a picture that is suspected to have been tampered with by Photoshop. " + inp

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

    outputs = tokenizer.decode(output_ids[0]).strip()
    outputs = outputs.replace("<s>", "").replace("</s>", "")

    with open(output_path, "w") as f:
        json.dump({"image": image_path, "outputs": outputs}, f)

    return outputs


def main(args):
    import random
    print("======== Loading Model Once ========")
    tokenizer, model, image_processor, context_len, DTG, model_name = DTE_FDM_init(args)
    print("======== Model Loaded ========")

    folders = [
               "/mnt/tank/scratch/dstoronkin/fakeshield_test/hwei_part1"
               ]
    for folder in folders:
        print(f"Processing {folder}:")
        image_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        image_files = random.sample(image_files, min(500, len(image_files)))

        real_count = 0
        fake_count = 0

        answers = []
        
        os.makedirs(args.output_path, exist_ok=True)

        for img_name in tqdm(image_files):
            image_path = os.path.join(folder, img_name)
            json_path = os.path.join(args.output_path, img_name + ".json")

            output_text = detect_single_image(
                args,
                tokenizer,
                model,
                image_processor,
                DTG,
                model_name,
                image_path,
                json_path
            )

            # === ПАРСИНГ ===
            label = not "has not been tampered with" in output_text.lower()
            
            if label:
                fake_count += 1
            else:
                real_count += 1
                
            answers.append([image_path, label])

        path = Path(folder)
        print(path.name)
        with open(f"{path.name}.txt", "w") as f:
            for item in answers:
                f.write(f"{item[0]} {int(item[1])}\n")
            f.write(f"Total images: {len(image_files)}\n")
            f.write(f"Predicted REAL: {real_count}\n")
            f.write(f"Predicted DEEPFAKE: {fake_count}\n")
        print("Total images:", len(image_files))
        print("Predicted REAL:", real_count)
        print("Predicted DEEPFAKE:", fake_count)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/tank/scratch/dstoronkin/weight/DTE-FDM")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--DTG-path", type=str, default="/mnt/tank/scratch/dstoronkin/weight/DTG.pth")
    parser.add_argument("--image-folder", type=str, default="/mnt/tank/scratch/dstoronkin/fakeshield_test/hwei_part1")
    parser.add_argument("--output-path", type=str, default="output_ovi")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(args.device)
    main(args)