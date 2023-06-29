This repo stores the code for [Explainable Verbal Reasoner Plus (EVR+): A Natural Language Reasoning Framework that Supports Diverse Compositional Reasoning](https://arxiv.org/abs/2305.00061), where a reasoning framework is proposed to flexibly compose language models to solve diverse compositional reasoning problems. 

# Download the SynthCompR Dataset
The dataset is available through Huggingface Hub. Repo link: [here](https://huggingface.co/zhengzhongliang/SynthCompR/tree/main)

# Repo Organization
## preliminary_experiments
 + data_generation: generate the 5 synthetic tasks in the paper. All tasks require compositional reasoning. The file ends with `_evr` are those used to generate the training data for EVR+.
 + experiments_e2e: has the scripts that fine-tune a UnifiedQA-T5-large model on the 5 tasks. The model is fine-tuned in a end-to-end manner.
 + experiments_evr: has the scripts that train and evaluate the EVR+ model.
   + The EVR+ interpreter: in the `evr_class` folder.
   + To train: use `train_evr.py`
   + To evaluate: use `eval_evr.py`

## prototype_tests
These are the tests that are used to make sure the interpreter works as expected. 
