import json
import re
import os

# === Path configuration ===
INPUT_FILE = '/path/to/action_grounding/input_data.json'    # Source file with IDs
OUTPUT_FILE = '/path/to/action_grounding_reverse/output_data.json'  # Output file
ERROR_LOG = 'error.log'             # Error log

# Try to import the progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, desc=""): return iterator

def extract_data(messages):
    """
    Extract the instruction and coordinates from the conversation history.
    """
    user_text = ""
    assist_text = ""

    # 1. Get the raw content of the User and Assistant turns
    for msg in messages:
        if msg['role'] == 'user':
            user_text = msg['content']
        elif msg['role'] == 'assistant':
            assist_text = msg['content']

    # 2. Extract the instruction via regex
    # Match the content after "## User Instruction :"
    instr_pattern = r"## User Instruction\s*:\s*(.+)$"
    instr_match = re.search(instr_pattern, user_text, re.DOTALL)

    if not instr_match:
        raise ValueError("Failed to extract User Instruction")
    instruction = instr_match.group(1).strip()

    # 3. Extract the coordinates via regex
    # Match the content inside <|box_start|>(x,y)<|box_end|>
    # Supports (x,y) or (x1,y1),(x2,y2)
    coord_pattern = r"<\|box_start\|>(.*?)<\|box_end\|>"
    coord_match = re.search(coord_pattern, assist_text)

    if not coord_match:
        raise ValueError("Failed to extract coordinates")

    raw_coords = coord_match.group(1).strip()

    # 4. Format the coordinates
    # If a single point (x,y) is extracted, the target format needs a point pair (x,y),(x,y)
    if raw_coords.count('(') == 1:
        formatted_coords = f"{raw_coords},{raw_coords}"
    else:
        formatted_coords = raw_coords

    return instruction, formatted_coords

def format_user_prompt(coords):
    """Generate the target User Prompt template."""
    return (
        "<image>You are specialized in GUI understanding and grounding.\n\n"
        "There are three types of tasks with different output requirements:\n"
        "1. Action grounding: Generate a grounding instruction starting with a verb (e.g., 'Click ...').\n"
        "2. Grounding bbox: Generate a descriptive caption for the region within the bounding box.\n"
        "3. Grounding point: Generate a descriptive caption for the element at the specific point.\n\n"
        "Instruction:\n"
        "Based on the provided task type ID and screen coordinates, generate the corresponding sentence.\n"
        "The output must correspond to one and only one element on the entire screen.\n\n"
        "Current Task Configuration:\n"
        f"- Coordinates: <|box_start|>{coords}<|box_end|>\n"
        "- Task Type ID: 1\n\n"
        "Please generate the result for the current task:"
    )

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    print("Processing...")

    # Read the source file
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    success_cnt = 0
    error_cnt = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out, \
         open(ERROR_LOG, 'w', encoding='utf-8') as f_err:

        f_out.write('[\n')

        for i, item in enumerate(tqdm(data, desc="Converting")):
            try:
                # Extract the core information
                instr, coords = extract_data(item.get('messages', []))

                # Build the new object
                new_item = {
                    "id": item.get('id'),  # keep the original ID
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant."
                        },
                        {
                            "role": "user",
                            "content": format_user_prompt(coords)
                        },
                        {
                            "role": "assistant",
                            "content": instr
                        }
                    ],
                    "images": item.get('images', []),
                    "task_type_id": 1
                }

                # Write out
                if success_cnt > 0:
                    f_out.write(',\n')
                json.dump(new_item, f_out, ensure_ascii=False, indent=4)
                success_cnt += 1

            except Exception as e:
                error_cnt += 1
                f_err.write(f"ID: {item.get('id', 'Unknown')} | Error: {str(e)}\n")

        f_out.write('\n]')

    print(f"\nDone!")
    print(f"Success: {success_cnt}")
    print(f"Failed: {error_cnt} (see {ERROR_LOG})")
    print(f"Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()