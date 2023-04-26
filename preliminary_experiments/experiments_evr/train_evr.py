import os
import torch
import numpy as np
import random

from absl import app
from absl import flags
from datetime import datetime

from torch.utils.data import DataLoader

from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils

from preliminary_experiments.experiments_evr.evr_dataset import TasksDataset, PadCollate
from preliminary_experiments.experiments_evr.evr_trainer import EVRTrainer


FLAGS = flags.FLAGS
_TASK_NAME = flags.DEFINE_string("task_name", "chaining_v0.1", "the task name, e.g., tree_search_v0.5")
_DU = flags.DEFINE_integer("du", 2, "the depth of the training data. DU2 means the data with depth up to 2.")
_N_TRAIN = flags.DEFINE_integer("n_train", 500, "number of training instances")
_MODEL_NAME = flags.DEFINE_string("model_name", "allenai/unifiedqa-t5-base", "model's name")
_GRAD_ACCU = flags.DEFINE_integer("grad_accu", 8, ("the gradient accumulation number. Batch size=2 and gradient "
                                                   "accumulation=8 works similarly as batch size=16"))
_LR = flags.DEFINE_float("lr", 1e-4, "the learning rate")
_MODEL_SEED = flags.DEFINE_integer("model_seed", 0, "the random seed to use to train the model")
_MACHINE_SWITCH = flags.DEFINE_string("machine_switch", "hpc", "which machine to use")
_MODEL_LOAD_PATH = flags.DEFINE_string("model_load_path", "None", "load a pre-trained model from a checkpoint")
_TRANSFER_MODEL_DATA_N_AMT = flags.DEFINE_string("transfer_model_data_n_amt", "None", ("the amount of transfer learning"
                                                                                       "data"))


class TrainEVR:

    def __init__(self):
        self.training_config = {
            "machine_switch": _MACHINE_SWITCH.value,
    
            "task_name": _TASK_NAME.value,
            "train_du": _DU.value,
            "n_train": _N_TRAIN.value,
            "model_name": _MODEL_NAME.value,
            "batch_size": 2,
            "grad_accu": _GRAD_ACCU.value,
    
            # The default learning rate (recommended) seems to be 1e-4
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/230
            "lr": _LR.value,
            "model_seed": _MODEL_SEED.value,
            "model_load_path": _MODEL_LOAD_PATH.value,
            "transfer_model_data_n_amt": _TRANSFER_MODEL_DATA_N_AMT.value,
            "n_epoch": 100,
            "num_workers": 1,
    
            # 2000 means to evaluate the model after training for 2000 batches (i.e., 4000 examples)
            "eval_every_k_batch": 2000,
            "patient_num": 10 if _N_TRAIN.value < 2000 else 5,
            "dev_ratio": 0.1,
    
            "n_test_ood_examples": 1000
        }
    
        self.task_name = _TASK_NAME.value
        self.du = _DU.value
        self.n_train = _N_TRAIN.value
        self.model_name = _MODEL_NAME.value
        self.grad_accu = _GRAD_ACCU.value
        self.lr = _LR.value
        self.batch_size = self.training_config["batch_size"]
    
        self.model_seed = _MODEL_SEED.value
        self.machine_switch = _MACHINE_SWITCH.value
        self.model_load_path = _MODEL_LOAD_PATH.value
        self.transfer_model_data_n_amt = _TRANSFER_MODEL_DATA_N_AMT.value
    
        self.n_epoch = self.training_config["n_epoch"]
        self.eval_every_k_batch = self.training_config["eval_every_k_batch"]
        self.patient_num = self.training_config["patient_num"]
        self.num_workers = self.training_config["num_workers"]
        self.dev_ratio = self.training_config["dev_ratio"]
    
        self.n_test_ood_examples = self.training_config["n_test_ood_examples"]
    
        self.save_folder_path = {
            "mac": "",
            "alix": "/home/zhengzhongliang/CLU_Projects/2022_IntermediateAnnotation/saved_models/",
            "hpc": "/xdisk/msurdeanu/zhengzhongliang/",
            "hpc_user": "/home/u15/zhengzhongliang/2022_IntermediateAnnotation/saved_models/"
        }
    
        self.model_save_name = {
            "t5-small": "t5small",
            "t5-large": "t5large",
            "allenai/unifiedqa-t5-small": "unifiedqa-t5-small",
            "allenai/unifiedqa-t5-base": "unifiedqa-t5-base",
            "allenai/unifiedqa-t5-large": "unifiedqa-t5-large"
        }
    
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
        # What information needs to be specified by the save folder path?
        # date, exp setting (e2e, e2e with explanation, evr), task name, transfer name and amt
        self.date_year_month_day = datetime.today().strftime('%Y%m%d')
        self.task_name_save = "-".join([ExpDatasetUtils.remove_dataset_version(t_n) for t_n in self.task_name.split("-")])
        self.exp_save_folder_path = self.save_folder_path[self.machine_switch] + self.date_year_month_day + \
                                    "_evr_" + self.task_name_save + \
                                    ("" if self.transfer_model_data_n_amt == "None"
                                     else "_" + self.transfer_model_data_n_amt) + "/"
    
        os.makedirs(self.exp_save_folder_path, exist_ok=True)
    
        # What information needs to be specified by the save file root name?
        # model_name, n_train, model_seed,
        self.save_file_root_name = self.model_save_name[self.model_name] + "_n_train_" + \
                                   str(self.n_train) + "_seed_" + str(self.model_seed) + \
                                   "_du_" + str(self.du)

    def sample_training_instances_by_pattern(self, training_instances_all_patterns, debug_flag=False):

        instances_by_pattern = {}
        for instance in training_instances_all_patterns:
            instance_pattern = f"{instance['task']}-{instance['pattern']}"
            if instance_pattern not in instances_by_pattern:
                instances_by_pattern[instance_pattern] = [instance]
            else:
                instances_by_pattern[instance_pattern].append(instance)

        patterns_to_oversample = ["chaining_v1.0_qa-3", "chaining_qa-3"]
        for pattern_to_oversample in patterns_to_oversample:
            if pattern_to_oversample in instances_by_pattern:
                instances_by_pattern[pattern_to_oversample] = instances_by_pattern[pattern_to_oversample] * 5

        training_instances_all_patterns = [i for pattern_instances in instances_by_pattern.values()
                                           for i in pattern_instances]

        if debug_flag:
            print("=" * 40)
            for pattern in instances_by_pattern:
                print(f"pattern: {pattern} {len(instances_by_pattern[pattern])}")
            print("total:", len(training_instances_all_patterns))
            input("-----")

        return training_instances_all_patterns

    def mix_training_instances(self, task_names):
        """Mix the instances from different tasks. E.g., mix chaining with cartesian."""

        task_names = task_names.split("-")
        task_names = [t_n for t_n in task_names if t_n != "" and not t_n.isspace()]

        mixed_instances = {}
        for task_name in task_names:
            instances_all_depth = ExpDatasetUtils.load_data(seed=self.model_seed,
                                                            n_train=self.n_train,
                                                            machine_switch=self.machine_switch,
                                                            data_pattern=task_name,
                                                            dev_ratio=self.dev_ratio)

            instances_all_depth = TasksDataset.get_evr_instances(instances_all_depth, task_name)

            for du in instances_all_depth.keys():

                # If the instances are mixed from different patterns, add some prefix to the pattern.
                if len(task_names) > 1:
                    for split in ["train", "dev", "test"]:
                        for instance in instances_all_depth[du][split]:
                            instance["task"] = f"{task_name}_{instance['task']}"

                if len(task_names) > 0 and task_name.startswith("cartesian_v"):
                    raw_du = du
                    mix_du = {3: 2, 4: 4}[raw_du]
                else:
                    raw_du = du
                    mix_du = raw_du

                if mix_du not in mixed_instances:
                    mixed_instances[mix_du] = {}

                for split in instances_all_depth[raw_du]:
                    if split not in mixed_instances[mix_du]:
                        mixed_instances[mix_du][split] = []

                    mixed_instances[mix_du][split].extend(instances_all_depth[raw_du][split])

        # For chaining qa 3 data, oversampling it so that it can be trained better.
        for du in mixed_instances:
            mixed_instances[du]["train"] = self.sample_training_instances_by_pattern(
                mixed_instances[du]["train"]
            )

        return mixed_instances

    @staticmethod
    def sample_test_examples(test_instances, sample_number):

        """
        This function is needed because we don't really need to evaluate the model on all of the test ood examples.
        That would be too time consuming. Instead, just try to sample some of them.
        :param test_instances:
        :param sample_number:
        :return:
        """

        if len(test_instances) > sample_number:
            return random.sample(test_instances, sample_number)
        else:
            return test_instances

    def train_evr_modules(self):

        evr_trainer = EVRTrainer(
            self.training_config,
            device=self.device,
            print_every=200,
            exp_save_folder_path=self.exp_save_folder_path,
            save_file_root_name=self.save_file_root_name
        )

        instances_all_depth = self.mix_training_instances(self.task_name)

        instances_train = instances_all_depth[self.du]["train"]
        instances_dev = instances_all_depth[self.du]["dev"]
        instances_test = instances_all_depth[self.du]["test"]

        # Sample the dev examples, so that we don't have to evaluate on all examples when there are too many examples.
        instances_dev = self.sample_test_examples(instances_dev, self.n_test_ood_examples)
        instances_test_ood = self.sample_test_examples(instances_all_depth[4]["test"], self.n_test_ood_examples)

        dataset_train = TasksDataset(instances_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers,
                                      collate_fn=PadCollate(
                                          tokenizer=evr_trainer.t5model.tokenizer,
                                          model_name=self.model_name),
                                      drop_last=False)

        dataset_dev = TasksDataset(instances_dev)
        dataloader_dev = DataLoader(dataset_dev, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers,
                                    collate_fn=PadCollate(
                                        tokenizer=evr_trainer.t5model.tokenizer,
                                        model_name=self.model_name),
                                    drop_last=False)

        dataset_test = TasksDataset(instances_test)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers,
                                     collate_fn=PadCollate(
                                         tokenizer=evr_trainer.t5model.tokenizer,
                                         model_name=self.model_name),
                                     drop_last=False)

        dataset_test_ood = TasksDataset(instances_test_ood)
        dataloader_test_ood = DataLoader(dataset_test_ood, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers,
                                         collate_fn=PadCollate(
                                             tokenizer=evr_trainer.t5model.tokenizer,
                                             model_name=self.model_name),
                                         drop_last=False)

        evr_trainer.train_and_eval(
            dataloader_train,
            dataloader_dev,
            dataloader_test,
            dataloader_test_ood,
            dataset_train,
            dataset_dev,
            dataset_test,
            dataset_test_ood
        )


def main(unused_argv):
    torch.manual_seed(_MODEL_SEED.value)
    random.seed(_MODEL_SEED.value)
    np.random.seed(_MODEL_SEED.value)

    print("*" * 40)
    print("DU:", _DU.value, " n train:", _N_TRAIN.value, " grad accu:", _GRAD_ACCU.value)
    print("lr:", _LR.value, " seed:", _MODEL_SEED.value, " data pattern:", _TASK_NAME.value)
    print("Cuda available:", torch.cuda.is_available())
    print("*" * 40)

    assert (_MODEL_NAME.value in [
        "t5-small", "t5-large",
        "allenai/unifiedqa-t5-small",
        "allenai/unifiedqa-t5-base",
        "allenai/unifiedqa-t5-large"])

    train_evr_instance = TrainEVR()
    train_evr_instance.train_evr_modules()


if __name__ == "__main__":
    app.run(main)
