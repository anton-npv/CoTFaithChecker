import json

def load_templates(template_path: str) -> dict:
    with open(template_path, 'r') as f:
        return json.load(f)

def construct_prompt(template: dict, question: dict, instruction_type: str) -> str:
    instruction = template[instruction_type]["cot"]
    prompt = instruction.format(question=question['q_str'])
    return prompt
