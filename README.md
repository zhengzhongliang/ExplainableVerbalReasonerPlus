This repo stores the code for [Explainable Verbal Reasoner Plus (EVR+): A Natural Language Reasoning Framework that Supports Diverse Compositional Reasoning](https://arxiv.org/abs/2305.00061), where a reasoning framework is proposed to flexibly compose language models to solve diverse compositional reasoning problems. 

# Download the SynthCompR Dataset
The dataset is available through Huggingface Hub. Repo link: [here](https://huggingface.co/datasets/zhengzhongliang/SynthCompR/tree/main)

# Requirements
`numpy==1.19.2`

`torch==1.13.1`

`transformers==4.1.0`

`absl`

# Run Scripts
## Before running
E.g., if the scripts are in the folder `/home/username/evr_plus/ExplainableVerbalReasonerPlus`, then first download the data to `/home/username/evr_plus/data/` using the link mentioned above. Assuming we are using the chaining dataset, then after downloading, the data path should be something like `/home/username/evr_plus/data/chaining_v1.0/chaining_data_du2.json`.

Also, please install the require python libraries and activate the corresponding virtual environment before running the scripts.

## Train and evaluate the end-to-end model
First run
`cd /home/username/evr_plus/ExplainableVerbalReasonerPlus`

An example script to train and evaluate the end-to-end model
```
python -m preliminary_experiments.experiments_end2end.train_t5e2e \
  --task_name="chaining_v1.0" --n_train=${N_TRAIN} \
  --batch_size=1 --grad_accu=16
```

## Train and evalate the EVR+ modules
```
python -m \
    preliminary_experiments.experiments_evr.train_evr \
    --task_name="cartesian_v1.0" --du=3 --n_train=2000 \
    --model_name="allenai/unifiedqa-t5-base"
```

## Run EVR+ inference with the trained modules
```
python -m preliminary_experiments.experiments_evr.eval_evr \
   --task_name=tree_search_v1.0 --du=4 --n_train=2000 \
   --eval_depth=${EVAL_DEPTH} \
   --neural_module_load_path=[some path] \
   --save_folder_path=[some path]
```
The EVAL_DPETH should be 0 to 4.

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
