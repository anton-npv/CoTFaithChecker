# CoT Unfaithfulness Investigations

This is Anton & Ida's Neel MATS 8.0 Sprint repo!

We are investigating CoT Unfaithfulness - why does a model write untruthful/unrelated/made-up stuff in its CoT?

This is our [presentation](https://docs.google.com/presentation/d/1fQMwDzENd_6xTv885ih_qPsL6hXHHb-zDT02CGNmpBk/edit?usp=sharing).

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


## Data

We used:
- MMLU  
- https://github.com/jettjaniak/chainscope/blob/main/chainscope/data/questions/datasets.md  
by the [Reasoning In The Wild](https://arxiv.org/pdf/2503.08679) paper


## Plan

We mapped out our strategy to be surveyed in this [pre-experimentation survey](https://forms.gle/qNeySxxxukUPkt9K9). Excited to compare with actual outcomes!

However: we do expect to dive in specifically where those experiments show curious patterns, and might drop other experiments.


## Implementation

Our research turned out to mainly focus on those experiments:  
- `a_confirm_posthoc` - initially re-doing the analysis of [Measuring Faithfulness](https://arxiv.org/pdf/2307.13702) for posthoc reasoning  
- `b_logprobts_analysis` - checking the certainty the model has on answer tokens throughout CoT generation  
- `c_cluster_analysis` - segmenting CoTs and clustering them into categories to analyse if individual parts are truthful  
- `e_confirm_xy_yx` - using the [Reasoning In The Wild](https://arxiv.org/pdf/2503.08679) setup to probe differences and track confidence  
- `i_cross_steer` - little test of steering with hint-faithfulness probe in xyyx-faithfulness setup  
- `j_probing` - probing faithful & unfaithful outcomes in hinted questions and steering to confirm  

