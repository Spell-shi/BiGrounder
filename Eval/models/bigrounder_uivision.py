import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from PIL import Image
import os
import re
import tempfile

# --- SFT prompt template ---
USER_PROMPT_TEMP = """You are a helpful assistant.
You are a GUI agent. You are given a task, with screenshots. You need to perform the next action to complete the task.
## Output Format
Action: ...
## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
rightClick(start_box='<|box_start|>(x1,y1)<|box_end|>')
doubleClick(start_box='<|box_start|>(x1,y1)<|box_end|>')
select(start_box='<|box_start|>(x1,y1)<|box_end|>') # To expand a dropdown menu or select an item from it.
type(content='', start_box='<|box_start|>(x1,y1)<|box_end|>') # First click on the textbox, and then type the content.
hotkey(key='')
stop(reason='') ## If the item can not found in the image, give the reason
scrolldown() ## If the all screenshot need to scroll down
scrollup() ## If the all screenshot need to scroll up
scrollmenu(start_box='<|box_start|>(x1,y1),(x2,y2)<|box_end|>') ## If the part of screenshot need to scroll and give the area need to scroll
finish()
"""

SYSTEM_PROMPT = "You are a helpful assistant."

def image_to_temp_filename(image):
    """Save a PIL image to a temporary file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    return temp_file.name

class BiGrounderModel():
    def load_model(self, model_name_or_path="Qwen/Qwen3-VL-2B-Instruct"):
        """Load the BiGrounder model (built on Qwen3-VL)."""
        print(f"Loading BiGrounder model from {model_name_or_path}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # Set generation parameters
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=True
            ).to_dict()
        except:
            self.generation_config = {}

        self.set_generation_config(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
        )

        print("BiGrounder model loaded successfully")

    def set_generation_config(self, **kwargs):
        """Set the generation config."""
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def _parse_coordinates(self, response):
        """
        Parse coordinates from the model output.
        Supported formats:
        1. (x,y) - single point (in the 0-1000 range)
        2. (x1,y1),(x2,y2) - region

        Returns:
            point: (x, y) normalized to 0-1, or None if absent
            area: (x1, y1, x2, y2) normalized to 0-1, or None if absent
        """
        # Pattern 1: match a region (x1,y1),(x2,y2)
        pattern_area = r"\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)"
        match_area = re.search(pattern_area, response)

        if match_area:
            x1 = int(match_area.group(1)) / 1000.0
            y1 = int(match_area.group(2)) / 1000.0
            x2 = int(match_area.group(3)) / 1000.0
            y2 = int(match_area.group(4)) / 1000.0

            # Compute the center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            return (center_x, center_y), (x1, y1, x2, y2)

        # Pattern 2: match a single point (x,y)
        pattern_point = r"\((\d+)\s*,\s*(\d+)\)"
        match_point = re.search(pattern_point, response)

        if match_point:
            x = int(match_point.group(1)) / 1000.0
            y = int(match_point.group(2)) / 1000.0
            return (x, y), None

        return None, None

    def _generate_response(self, instruction, image):
        """
        Internal generation function.

        Args:
            instruction: the task instruction
            image: a PIL Image object or an image path

        Returns:
            response: the text output of the model
            resized_image: the resized image (for subsequent processing)
        """
        # Handle the image input
        if isinstance(image, str):
            assert os.path.exists(image) and os.path.isfile(image), "Invalid input image path."
            pil_image = Image.open(image).convert('RGB')
        else:
            pil_image = image.convert('RGB')

        # 1. Image preprocessing - use smart_resize
        resized_height, resized_width = smart_resize(
            pil_image.height,
            pil_image.width,
            factor=32,
            min_pixels=32 * 32,
            max_pixels=12845056,
        )
        resized_image = pil_image.resize((resized_width, resized_height))

        # 2. Build messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": USER_PROMPT_TEMP + f"\n## User Instruction: {instruction}"}
                ]
            }
        ]

        # 3. Prepare model inputs
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_input],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 4. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.generation_config)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # skip_special_tokens=True strips the special markers
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response, resized_image

    def ground_only_positive(self, instruction, image):
        """
        Element task - locate the element and return the click coordinates.

        Args:
            instruction: the task instruction
            image: a PIL Image object or an image path

        Returns:
            result_dict: a dict containing the prediction results
        """
        # Generate the response
        response, resized_image = self._generate_response(instruction, image)

        print(f"[Element] Raw Response: {response}")

        result_dict = {
            "result": "positive",
            "format": "point",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse coordinates
        try:
            point, area = self._parse_coordinates(response)

            if point is not None:
                # The coordinates are already normalized (0-1 range)
                result_dict["point"] = list(point)
                # Generate a pseudo bbox for evaluation
                result_dict["bbox"] = [point[0], point[1], point[0], point[1]]
                print(f"[Element] Parsed Point: {point}")
            else:
                # Check whether this is a stop instruction
                response_lower = response.lower()
                if "stop" in response_lower or "finish" in response_lower:
                    print("[Element] Action is stop/finish")
                    result_dict["result"] = "negative"
                else:
                    print("[Element] No coordinates found")
                    result_dict["result"] = "negative"

        except Exception as e:
            print(f"[Element] Parse Error: {e}")
            result_dict["result"] = "error"

        return result_dict

    def layout_gen(self, name, explanation, image):
        """
        Layout task - generate the bounding box of an element.

        Args:
            name: the element name
            explanation: the element explanation
            image: a PIL Image object or an image path

        Returns:
            result_dict: a dict containing the prediction results
        """
        # Build the instruction for the Layout task
        task_instruction = f"Locate the element '{name}'. Explanation: {explanation}."

        # Generate the response
        response, resized_image = self._generate_response(task_instruction, image)

        print(f"[Layout] Raw Response: {response}")

        result_dict = {
            "result": "positive",
            "format": "bbox",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse coordinates
        try:
            point, area = self._parse_coordinates(response)

            # The Layout task prefers the region box
            if area is not None:
                # The region coordinates are already normalized
                result_dict["bbox"] = list(area)
                print(f"[Layout] Parsed BBox: {area}")
            elif point is not None:
                # If only a point is available, create a small box
                result_dict["bbox"] = [point[0], point[1], point[0], point[1]]
                print(f"[Layout] Parsed Point (as BBox): {point}")
            else:
                print("[Layout] No coordinates found")
                result_dict["result"] = "negative"

        except Exception as e:
            print(f"[Layout] Parse Error: {e}")
            result_dict["result"] = "error"

        return result_dict
