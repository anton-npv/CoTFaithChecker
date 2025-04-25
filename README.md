# CoT Unfaithfulness Investigations

This is Anton & Ida's Neel MATS 8.0 Sprint repo!

We are investigating CoT Unfaithfulness - why does a model write untruthful/unrelated/made-up stuff in its CoT?

### We define:

- Truthfulness - a given step in the output CoT is actually happening in the internal processes  
- Monitorability - does a given step have a causal impact on the final answer?  
- Completeness - the model does not skip any internal steps in the output CoT  

## Structure

```
.
├── README.md
├── a_confirm_posthoc
│   ├── eval
│   ├── main
│   ├── outputs
│   └── utils
├── b_logprobs_analysis
│   ├── main
│   └── outputs
├── c_ablate_downstream
│   ├── eval
│   ├── main
│   └── outputs
├── d_attention_hint
│   ├── attention_check_500.json
│   └── attention_checker.py
├── e...
├── data
│   ├── gsm8k
│   ├── mmlu
│   └── src
└── notebooks
    ├── a_confirm_posthoc.ipynb
    ├── a_faithfulness_check.ipynb
    ├── a_sanity_check.ipynb
    ├── b_logprobs_anaylsis.ipynb
    ├── c_ablate_downstream.ipynb
    ├── d_attention_hint.ipynb
    └── e...
```

(note: sudo apt update && sudo apt install tree)

## Data

- MMLU  
- https://github.com/jettjaniak/chainscope/blob/main/chainscope/data/questions/datasets.md  
