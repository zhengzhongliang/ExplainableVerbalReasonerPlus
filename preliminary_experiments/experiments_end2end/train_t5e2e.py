from absl import app
from absl import flags

import os
import numpy as np
import torch
import random
import json
from datetime import datetime
from torch.utils.data import DataLoader

from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils
from preliminary_experiments.data_generation.dataset_end2end_utils import DatasetEnd2endUtils
from preliminary_experiments.experiments_end2end.t5e2e_dataset import DatasetT5E2E, PadCollate
from preliminary_experiments.experiments_end2end.t5e2e_trainer import T5End2EndTrainer

FLAGS = flags.FLAGS
_TASK_NAME = flags.DEFINE_string("task_name", "chaining_v0.1", "the task name, e.g., tree_search_v0.5")
_DU = flags.DEFINE_integer("du", 2, "the depth of the training data. DU2 means the data with depth up to 2.")
_N_TRAIN = flags.DEFINE_integer("n_train", 500, "number of training instances")
_MODEL_NAME = flags.DEFINE_string("model_name", "allenai/unifiedqa-t5-large", "model's name")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 2, "Batch size.")
_GRAD_ACCU = flags.DEFINE_integer("grad_accu", 8, ("the gradient accumulation number. Batch size=2 and gradient "
                                                   "accumulation=8 works similarly as batch size=16"))
_LR = flags.DEFINE_float("lr", 1e-4, "the learning rate")
_MODEL_SEED = flags.DEFINE_integer("model_seed", 0, "the random seed to use to train the model")
_MACHINE_SWITCH = flags.DEFINE_string("machine_switch", "hpc", "which machine to use")
_MODEL_LOAD_PATH = flags.DEFINE_string("model_load_path", "None", "load a pre-trained model from a checkpoint")
_TRANSFER_MODEL_DATA_N_AMT = flags.DEFINE_string("transfer_model_data_n_amt", "None", ("the amount of transfer learning"
                                                                                       "data"))


class TrainT5E2E:

    def __init__(self):
        self.parsed_tasks = ExpDatasetUtils.parse_tasks(_TASK_NAME.value)

        self.model_config = {
            "model_gen_len": 400,
        }

        self.training_config = {
            "machine_switch": _MACHINE_SWITCH.value,

            "task_name": _TASK_NAME.value,
            "train_du": _DU.value,
            "n_train": _N_TRAIN.value,
            "model_name": _MODEL_NAME.value,
            "batch_size": _BATCH_SIZE.value,
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
            "dev_ratio": 0.1
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
            "allenai/unifiedqa-t5-large": "unifiedqa-t5-large"
        }

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # What information needs to be specified by the save folder path?
        # date, exp setting (e2e, e2e with explanation, evr), task name, transfer name and amt
        self.task_name_save = "-".join([
            ExpDatasetUtils.remove_dataset_version(t_n) for t_n in self.task_name.split("-")])
        self.date_year_month_day = datetime.today().strftime('%Y%m%d')
        self.exp_save_folder_path = (
            f"{self.save_folder_path[self.machine_switch]}{self.date_year_month_day}_t5e2e_{self.task_name_save}/"
        ) if self.transfer_model_data_n_amt == "None" else (
            f"{self.save_folder_path[self.machine_switch]}{self.date_year_month_day}_t5e2e_{self.task_name_save}_"
            f"{self.transfer_model_data_n_amt}/"
        )

        os.makedirs(self.exp_save_folder_path, exist_ok=True)

        # What information needs to be specified by the save file root name?
        # model_name, n_train, model_seed,
        self.save_file_root_name = (
            f"{self.model_save_name[self.model_name]}_n_train_{self.n_train}_seed_{self.model_seed}"
        )

        print("*" * 40)
        print("model config:", json.dumps(self.model_config, indent=2))
        print("exp config:", json.dumps(self.training_config, indent=2))
        print("Cuda available:", torch.cuda.is_available())
        print("*" * 40)

    def mix_training_instances(self, task_names):

        print("*" * 40)
        print(f"start loading data for {task_names} ...")
        print("*" * 40)

        task_names = task_names.split("-")
        task_names = [t_n for t_n in task_names if t_n != "" and not t_n.isspace()]

        mixed_instances = {}
        for task_name in task_names:
            instances_all_depth = ExpDatasetUtils.load_data(seed=self.model_seed,
                                                            n_train=self.n_train,
                                                            machine_switch=self.machine_switch,
                                                            data_pattern=task_name,
                                                            dev_ratio=self.dev_ratio)

            instances_all_depth = DatasetEnd2endUtils.convert_instances(
                instances_all_depth, task_name, chunk_size=3)

            for du in instances_all_depth.keys():

                if len(task_names) > 1 and task_name.startswith("cartesian_v"):
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

        return mixed_instances

    def train_t5e2e(self):

        # Load the trainer:
        t5e2e_trainer = T5End2EndTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            exp_save_folder_path=self.exp_save_folder_path,
            save_file_root_name=self.save_file_root_name,
        )

        # Load the data
        instances_all_depth = self.mix_training_instances(self.task_name)

        instances_train = instances_all_depth[self.du]["train"]
        instances_dev = instances_all_depth[self.du]["dev"]
        instances_test = instances_all_depth[self.du]["test"]

        instances_test_ood = instances_all_depth[4]["test"]

        print("*" * 40)
        print(f"Num training instances: {len(instances_train)}")
        print(f"Num dev instances: {len(instances_dev)}")
        print(f"Num test ood instances: {len(instances_test_ood)}")
        print("*" * 40)

        dataset_train = DatasetT5E2E(instances_train)
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers,
                                      collate_fn=PadCollate(
                                          tokenizer=t5e2e_trainer.t5model.tokenizer,
                                          model_name=self.model_name),
                                      drop_last=False)

        dataset_dev = DatasetT5E2E(instances_dev)
        dataloader_dev = DataLoader(dataset_dev, batch_size=self.batch_size, shuffle=False,
                                    num_workers=self.num_workers,
                                    collate_fn=PadCollate(
                                        tokenizer=t5e2e_trainer.t5model.tokenizer,
                                        model_name=self.model_name),
                                    drop_last=False)

        dataset_test = DatasetT5E2E(instances_test)
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers,
                                     collate_fn=PadCollate(
                                         tokenizer=t5e2e_trainer.t5model.tokenizer,
                                         model_name=self.model_name),
                                     drop_last=False)

        dataset_test_ood = DatasetT5E2E(instances_test_ood)
        dataloader_test_ood = DataLoader(dataset_test_ood, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers,
                                         collate_fn=PadCollate(
                                             tokenizer=t5e2e_trainer.t5model.tokenizer,
                                             model_name=self.model_name),
                                         drop_last=False)

        t5e2e_trainer.train_and_eval(
            dataloader_train,
            dataloader_dev,
            dataloader_test,
            dataloader_test_ood,
            instances_train,
            instances_dev,
            instances_test,
            instances_test_ood,
        )


def main(unused_argv):
    torch.manual_seed(_MODEL_SEED.value)
    random.seed(_MODEL_SEED.value)
    np.random.seed(_MODEL_SEED.value)

    assert (_MODEL_NAME.value in ["t5-small", "t5-large", "allenai/unifiedqa-t5-small", "allenai/unifiedqa-t5-large"])

    train_e2d_instance = TrainT5E2E()
    train_e2d_instance.train_t5e2e()


if __name__ == "__main__":
    app.run(main)

