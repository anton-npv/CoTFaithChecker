import json
from pathlib import Path
from typing import List
from cot_faithfulness.logging_utils import init_logger

__all__ = ["PromptBuilder"]

logger = init_logger(Path.cwd() / "logs", "prompt_builder")


class PromptBuilder:
    """
    Builds prompts from the Chainscope instruction templates.
    """

    def __init__(
        self,
        template_path: Path = Path("data/chainscope/templates/instructions.json"),
        style: str = "instr-v0",
        mode: str = "cot",  # "cot" | "direct" | "open_ended_cot"
    ):
        with template_path.open() as f:
            self.templates = json.load(f)
        self.template = self.templates[style][mode]
        logger.info(f"PromptBuilder initialised with style={style}, mode={mode}")

    def build(self, question: str) -> str:
        return self.template.replace("{question}", question)

    def batch(self, questions: List[str]) -> List[str]:
        return [self.build(q) for q in questions]
