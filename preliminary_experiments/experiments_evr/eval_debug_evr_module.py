import torch
import sys
from torch.utils.data import DataLoader
import os

from absl import flags
from absl import app

from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils
from preliminary_experiments.utils.experiment_metric_utils import ExpMetricUtils

from preliminary_experiments.experiments_evr.evr_class.evr_backbone_lm import T5ModelEVR
from preliminary_experiments.experiments_evr.evr_trainer import EVRTrainer
from preliminary_experiments.experiments_evr.evr_dataset import TasksDataset, PadCollate

FLAGS = flags.FLAGS
_TASK_NAME = flags.DEFINE_string("task_name", "cartesian_v1.0", "The task to evaluate.")
_DU_TRAIN = flags.DEFINE_integer("du_train", 3, "Data depth used in training")
_DU_TEST = flags.DEFINE_integer("du_test", 4, "Data depth used for test.")
_N_TRAIN = flags.DEFINE_integer("n_train", 2000, "Number of examples used in training.")
_MODEL_NAME = flags.DEFINE_string("model_name", "allenai/unifiedqa-t5-base", "The name of the model used.")
_GRAD_ACCU = flags.DEFINE_integer("grad_accu", 8, "Gradient accumulation number.")
_LR = flags.DEFINE_float("lr", 1e-4, "Learning rate.")
_MODEL_SEED = flags.DEFINE_integer("model_seed", 0, "The random seed used to train the model.")
_MACHINE_SWITCH = flags.DEFINE_string("machine_switch", "hpc", "On what machine the script will be executed.")
_EVAL_PATTERN = flags.DEFINE_string("eval_pattern", "generate_program-1", "On what module data the model is evalauted.")
_EVAL_DEPTH = flags.DEFINE_integer("eval_depth", 4, "Evaluation data depth")
_MODEL_FOLDER_PATH = flags.DEFINE_string("model_folder_path", "None", "From which folder the model will be loaded.")
_MODEL_NAME_ROOT = flags.DEFINE_string("model_name_root", "None", "The actual model's name.")

print("*" * 40)
print("cuda available:", torch.cuda.is_available())
print("*" * 40)


class EVRModuleEvaluator:

    def __init__(self,
                 task_name,
                 du_train,
                 du_test,
                 n_train,
                 model_name,
                 batch_size,
                 lr,
                 grad_accu_num,
                 model_seed,
                 machine_switch,
                 eval_pattern,
                 eval_depth
                 ):

        self.task_name = task_name
        self.du_train = du_train
        self.du_test = du_test
        self.n_train = n_train
        self.model_name = model_name
        self.batch_size = batch_size
        self.lr = lr
        self.grad_accu_num = grad_accu_num
        self.model_seed = model_seed

        self.machine_switch = machine_switch

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

        self.eval_pattern = eval_pattern
        self.eval_depth = eval_depth

    def load_and_eval(self, split="test", debug_flag=False):

        assert split in ["train", "dev", "test"]

        # Load the trained model
        exp_save_folder_path = _MODEL_FOLDER_PATH.value

        save_file_root_name = _MODEL_NAME_ROOT.value

        t5model = T5ModelEVR(
            model_name=self.model_name,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            model_load_path=os.path.join(exp_save_folder_path, save_file_root_name),
        )

        # Load the instances
        instances_all_du = ExpDatasetUtils.load_data(seed=self.model_seed,
                                                     n_train=self.n_train,
                                                     machine_switch=self.machine_switch,
                                                     data_pattern=self.task_name,
                                                     dev_ratio=0.1)
        instances_all = TasksDataset.get_evr_instances(instances_all_du, self.task_name)[self.du_test]["test"]
        instances_eval = [instance for instance in instances_all
                          if instance["depth"] == int(self.eval_depth) and
                          instance["task"] == self.eval_pattern.split("-")[0] and
                          instance["pattern"] == int(self.eval_pattern.split("-")[1])]

        print("=" * 40)

        # Evaluate only the instances with certain depth
        dataset_eval = TasksDataset(instances_eval)
        dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=2,
                                     collate_fn=PadCollate(tokenizer=t5model.tokenizer, model_name=self.model_name),
                                     drop_last=False)

        # Do the evaluation and debug
        eval_acc, input_text_all, pred_text_all, target_text_all, hit_by_task_n_depth, ids_all = \
            EVRTrainer.eval_epoch_public(t5model, dataloader_eval, instances_eval, debug_flag=debug_flag)

        print("du train:", self.du_train, " du test:", self.du_test,
              " depth:", self.eval_depth, " num of instances:", len(instances_eval))
        print("split ", split, " acc:", eval_acc)

        if debug_flag:
            for idx in range(len(input_text_all)):
                print("=" * 40)
                print("input:")
                print(input_text_all[idx])
                print("pred:")
                print(pred_text_all[idx])
                print("target:")
                print(target_text_all[idx])
                print("target len:")
                print(len(t5model.tokenizer(target_text_all)["input_ids"]))
                print("hit?", pred_text_all[idx] == target_text_all[idx])
                input("-" * 40)


def main(unused_argv):
    evr_evaluator = EVRModuleEvaluator(
        task_name=_TASK_NAME.value,
        du_train=_DU_TRAIN.value,
        du_test=_DU_TEST.value,
        n_train=_N_TRAIN.value,
        model_name=_MODEL_NAME.value,
        batch_size=2,
        lr=_LR.value,
        grad_accu_num=_GRAD_ACCU.value,
        model_seed=_MODEL_SEED.value,
        machine_switch=_MACHINE_SWITCH.value,
        eval_pattern=_EVAL_PATTERN.value,
        eval_depth=_EVAL_DEPTH.value
    )

    evr_evaluator.load_and_eval(debug_flag=True)


if __name__ == "__main__":
    app.run(main)