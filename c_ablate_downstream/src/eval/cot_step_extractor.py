"""
Ask a verifier LLM to identify *three key reasoning steps* in each
full-length completion and return them as free-text strings.
"""
import os, json, tqdm
from typing import List, Dict
from google import genai            # already used elsewhere
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def _prompt(completion: str) -> str:
    return f"""You are analysing a chain-of-thought.

Return exactly a JSON list with **three** short quotations (<= 25 words each)
that represent the *three most important reasoning steps* that led from the
information in the question to the final answer. Use the *verbatim* text, no
added commentary.

If the CoT is shorter than three reasoning steps, repeat the last step so the
list still has length three.

Completion:
{completion}"""

def extract_steps(completions_path: str,
                  out_path: str) -> None:
    comps = json.load(open(completions_path))
    out: List[Dict] = []
    for c in tqdm.tqdm(comps, desc="step-extract"):
        resp = client.models.generate_content(
            model="gemini-2.0-pro",
            contents=_prompt(c["completion"]),
            config={"response_mime_type": "application/json"}
        )
        out.append({"question_id": c["question_id"],
                    "steps": resp.text if isinstance(resp, str) else resp.parsed})
    json.dump(out, open(out_path,"w"), indent=2)
