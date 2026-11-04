# Data Construction

Scripts in this directory construct **reverse** training data from existing
forward grounding data. We take action grounding as a worked example.

## Motivation

Standard GUI grounding data is **forward**: given an instruction, the model
predicts screen coordinates.

```
instruction  ──►  coordinates
```

To strengthen the model's bidirectional understanding of the
instruction↔coordinate mapping, we build the **reverse** task from the same
data: given the coordinates (and a task-type ID), the model generates the
corresponding instruction.

```
coordinates  ──►  instruction
```

The two directions share the same underlying (image, instruction, coordinate)
triples, so no new annotation is needed — the reverse set is derived
automatically from the forward set.

## Example: `reverse_action_grounding.py`

This script converts forward **action grounding** samples into reverse ones.

**Input** — forward samples where the user turn carries the instruction and the
assistant turn carries the target coordinates:

- User turn contains `## User Instruction : <instruction>`
- Assistant turn contains `<|box_start|>(x,y)<|box_end|>`
  (single point `(x,y)` or a pair `(x1,y1),(x2,y2)`)

**Output** — reverse samples where the roles are swapped around the coordinate:

- System: `You are a helpful assistant.`
- User: a task prompt that specifies the **coordinates** and the **task-type ID**
- Assistant: the **instruction** to be generated

Each sample keeps its original `id` and `images`, and is tagged with
`task_type_id: 1` (action grounding).

### Processing steps

1. Read the forward JSON list.
2. For each sample, extract the instruction and the coordinates via regex.
3. Normalize a single point `(x,y)` into a point pair `(x,y),(x,y)` so all
   samples share one coordinate format.
4. Emit the reversed sample using the prompt template from `format_user_prompt`.
5. Log any sample that fails extraction to `error.log` and continue.

### Usage

Edit the `INPUT_FILE` and `OUTPUT_FILE` paths at the top of the script, then run:

```bash
python reverse_action_grounding.py
```

The script prints the success/failure counts and writes the reversed dataset to
`OUTPUT_FILE`. Samples that could not be parsed are recorded in `error.log`.

## Task-type IDs

The reverse prompt is designed to cover three task types (only type 1 is built
by this example script):

| ID | Task | Reverse output |
|----|------|----------------|
| 1  | Action grounding | An instruction starting with a verb (e.g. `Click ...`) |
| 2  | Grounding bbox    | A descriptive caption for the region in the bounding box |
| 3  | Grounding point   | A descriptive caption for the element at the point |

Other task types follow the same forward→reverse construction pattern shown
here.
