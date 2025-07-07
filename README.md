The baseline of this project is the replication of the "Reasoning Models Dont Always Say What They Think" paper by Anthropic. In this paper, the authors conduct the following experiment to show that reasoning LLMs often produce unfaithful completions:

- give the model the same question in 2 different setups. Setup 1: provide only the MCQ question and generate the response. Setup 2: provide the question together with a hint (they experimented with different hint types, eg Sycophancy or Unethical Information) eg "I think that the answer is A"; generate the completion.

- repeat this for all the MCQ questions in the dataset

- check if the model's final answer changes from the setup 1 in setup 2 to the provided hint

- if the answer has changed, use a verifier LLM to check if the model verbalises the hint in its chain of thought

- if the chain of thought does verbalise the hint it is considered to be faithful; if it doesn't - then the completion is unfaithful


I have replicated this part of the paper and saved the data into the data/mmlu_new folder. The furhter folder strcutre contains model name, as well as the types of hint used (eg none (no hint) or sycophancy). Note that the data generation pipeline can be found in a_confirm_posthoc/parallelization/driver.py

The data is broken down into multiple files:

1. completions_with____.json - this contains the following fields: question_id and completion. Note that completion is actually both the user prompt and the model completion bunched together.

2. verification_with____.json - this contains the question_id and verified_answer which is the final answer the model provided. The verified answer is obtained by running a separate LLM model on the completion generated.

3. switch_analysis_with___.json - this compares the completions generated without and with the hint and looks at whether the answer has switched to the hint. It contains the following fields:
    - question_id
    - switched - boolean
    - to_intended_hint - boolean - checks whether the answer with the hint matches the hint provided
    - hint_option - str - this is actually the hint option provided by the user (eg "C")
    - is_correct_option - boolean - whether the hitn provided is the gorund truth

4. hint_verification_with___.json - this is prdoced by a separate verfifier LLM that only looks at the questions that have switched to the intended hint and contains the following fields:
    - question_id - int
    - mentions_hint - boolean - whether the CoT contains ANY mention of the hint
    - uses_hint_only_for_verification - boolean - checks whether the CoT only mentions the hint at the end once it has already independetly arrived at the answer
    - depends_on_hint - boolean - actually uses the hint in the CoT to arrive at the final anseer
    - explanation - str - verifier model generated explanation for uses_hint_only_for_verification and depends_on_hint, and verbalizes_hint
    - verbalizes_hint - boolean - thats the main bool variable which essentially stands for whether a given completion is faithful or not
    quartiles - list[int] - quartiles in which CoT refers to the hint