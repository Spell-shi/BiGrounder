import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from PIL import Image
import os
import re
import json

# --- BiGrounder prompt template ---
USER_PROMPT_TEMP = """You are a helpful assistant,
    'You are a GUI agent. You are given a task, with screenshots. You need to perform the next action to complete the task. '
    '## Output Format '
    'Action: ... '
    '## Action Space '
    "click(start_box='<|box_start|>(x1,y1)<|box_end|>') "
    "rightClick(start_box='<|box_start|>(x1,y1)<|box_end|>')"
    "doubleClick(start_box='<|box_start|>(x1,y1)<|box_end|>') "
    "select(start_box='<|box_start|>(x1,y1)<|box_end|>') # To expand a dropdown menu or select an item from it. "
    "type(content='', start_box='<|box_start|>(x1,y1)<|box_end|>') # First click on the textbox, and then type the content. "
    "hotkey(key='') "
    "stop(reason='') ## If the item can not found in the image, give the reason "
    "scrolldown() ## If the all screenshot need to scroll down "
    "scrollup() ## If the all screenshot need to scroll up "
    "scrollmenu(start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>') ## If the part of screenshot need to scroll and give the area need to scroll "
    "finish() "
"""
SYSTEM_PROMPT = "You are a helpful assistant."

class BiGrounderModel():
    def load_model(self, model_name_or_path="Qwen/Qwen3-VL-2B-Instruct"):
        # Load the BiGrounder model with AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Set generation parameters
        try:
            self.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True).to_dict()
        except:
            self.generation_config = {}

        self.set_generation_config(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,  # adjust to the SFT output length
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # 1. Image preprocessing
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=32,
            min_pixels=32 * 32,
            max_pixels=12845056,
        )
        # print(f"Resized image size: {resized_width}x{resized_height}")
        resized_image = image.resize((resized_width, resized_height))

        # 2. Build the prompt
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": USER_PROMPT_TEMP + f"## User Instruction : {instruction}"}
                ]
            }
        ]

        # 3. Prepare model inputs
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 4. Generate
        generated_ids = self.model.generate(**inputs, **self.generation_config)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Note: skip_special_tokens=True strips <|box_start|> etc., so the output becomes (300,142)
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"Raw Response: {response}")

        result_dict = {
            "result": "positive",
            "format": "point",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # 5. Parsing logic
        try:
            # Regex: directly match (number,number)
            # Handles (300,142) or (300, 142)
            # This matches start_box='(300,142)' in the raw response
            pattern = r"\((\d+)\s*,\s*(\d+)\)"

            match = re.search(pattern, response)

            if match:
                x_int = int(match.group(1))
                y_int = int(match.group(2))
                print(f"Parsed Point: {x_int}, {y_int}")

                # Normalize: the model outputs integers in 0-1000, divide by 1000 to get 0-1 floats
                result_dict["point"] = [x_int / 1000.0, y_int / 1000.0]
            else:
                # Check whether this is a stop instruction
                if "stop" in response.lower() or "finish" in response.lower():
                    print("Action is stop/finish")
                    result_dict["result"] = "negative"  # or set to another value per the eval set
                else:
                    print("No coordinates found")
                    result_dict["result"] = "negative"

        except Exception as e:
            print(f"Parse Error: {e}")
            result_dict["result"] = "error"

        return result_dict
