import numpy as np
import json
import torch

from absl import app
from absl import flags

from torch.utils.data import DataLoader

from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils
from preliminary_experiments.experiments_end2end.t5e2e_class import T5Model
from preliminary_experiments.experiments_end2end.t5e2e_dataset import DatasetT5E2E, PadCollate
from preliminary_experiments.experiments_end2end.t5e2e_trainer import T5End2EndTrainer
from preliminary_experiments.utils.experiment_metric_utils import ExpMetricUtils


FLAGS = flags.FLAGS
_DEBUG_FLAG = flags.DEFINE_boolean("debug_flag", True, "Whether to use debug mode to print all info.")
_DATA_PATTERN = flags.DEFINE_string("data_pattern", "chaining_v0.3", "the data pattern and version.")
_DU = flags.DEFINE_integer("du", 4, "The max data depth.")
_N_TRAIN = flags.DEFINE_integer("n_train", 500, "Number of training instances.")
_SEED = flags.DEFINE_integer("seed", 0, "The random seed to use.")
_MODEL_NAME = flags.DEFINE_string("model_name", "allenai/unifiedqa-t5-large", "The model to use for evaluation.")
_MODEL_LOAD_PATH = flags.DEFINE_string("model_load_path", "", "The path to load the model from.")


class EvalDebugT5E2E:

    @classmethod
    def load_and_eval(cls, depths=range(6), split="test", debug_flag=False):

        assert split in ["train", "dev", "test"]

        # Load the trained model
        t5model = T5Model(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                          model_name=_MODEL_NAME.value,
                          model_load_path=_MODEL_LOAD_PATH.value,
                          transfer_model_data_n_amt="None"
                          )

        # Load the instances
        instances_all_du = ExpDatasetUtils.load_data(seed=_SEED.value,
                                                     n_train=_N_TRAIN.value,
                                                     data_pattern=_DATA_PATTERN.value,
                                                     dev_ratio=0.1)

        file_for_error_analysis = {d: {} for d in depths}
        for depth in depths:
            # Evaluate only the instances with certain depth
            instances_eval = [instance for instance in instances_all_du[_DU.value][split] if instance["depth"] == depth]
            dataset_eval = DatasetT5E2E(instances_eval)
            dataloader_eval = DataLoader(dataset_eval, batch_size=4, shuffle=False, num_workers=2,
                                         collate_fn=PadCollate(tokenizer=t5model.tokenizer,
                                                               model_name=_MODEL_NAME.value),
                                         drop_last=False)

            # Do the evaluation and debug
            acc, input_text_all, pred_text_all, target_text_all, hit_by_depth, ids_all = \
                T5End2EndTrainer.eval_epoch_public(
                    t5model, dataloader_eval, instances_eval, task_name=_DATA_PATTERN.value, debug_flag=False)

            print("DU:", _DU.value, " depth:", depth, " num of instances:", len(instances_eval))
            print("split ", split, " acc:", acc)

            # This is to figure out the distribution of accuracy on different labels:
            acc_distrib_dict = {"Yes": [], "No": []}
            for idx in range(len(pred_text_all)):
                pred = pred_text_all[idx]
                target = target_text_all[idx]

                if pred == target:
                    acc_distrib_dict[target].append(1)
                else:
                    acc_distrib_dict[target].append(0)

            for l in acc_distrib_dict.keys():
                print(l, np.mean(acc_distrib_dict[l]), len(acc_distrib_dict[l]))
            print("=" * 40)

            if debug_flag:
                for idx in range(len(input_text_all)):
                    print("=" * 40)
                    print(input_text_all[idx])
                    print(pred_text_all[idx])
                    print(target_text_all[idx])
                    print("hit?", pred_text_all[idx] == target_text_all[idx])
                    input("-" * 40)

            file_for_error_analysis[depth] = {
                "instances": instances_eval,
                "pred_text": pred_text_all,
                "target_text": target_text_all
            }

        # with open(cls.save_folder_path[machine_switch] + exp_folder_name + "predictions.json", "w") as handle:
        #     json.dump(file_for_error_analysis, handle)

    @classmethod
    def load_and_eval_contrastive(cls, depths=(0, 1, 2, 3, 4, 5), split="test", machine_switch="alix",
                                  debug_flag=False):
        '''
        This function evaluates the model with only part of the context. This way we can know what the model learns to
        use make the predictions.
        :param depth:
        :param split:
        :param machine_switch:
        :param debug_flag:
        :return:
        '''

        assert split in ["dev", "test"]

        t5model = T5Model(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                          model_name=_MODEL_NAME.value,
                          model_load_path=_MODEL_LOAD_PATH.value,
                          transfer_model_data_n_amt="None"
                          )

        # Load the instances
        instances_all_du = ExpDatasetUtils.load_data(seed=_SEED.value,
                                                     n_train=_N_TRAIN.value,
                                                     data_pattern=_DATA_PATTERN.value,
                                                     dev_ratio=0.1)

        t5model.model.eval()

        file_for_error_analysis = {d: {} for d in depths}
        for depth in depths:

            instances_eval = [instance for instance in instances_all_du[_DU.value][split] if instance["depth"] == depth]

            input_text_all = []
            pred_text_all = []
            target_text_all = []
            context_length = []
            for sample in instances_eval:
                context_length.append(len(sample["context_list"]))
                context_string = " ".join(sample["context_list"])

                batch = {}
                batch["input_text"] = [context_string + " " + sample["question_string"]]
                batch["input"] = t5model.tokenizer(batch["input_text"],
                                                   return_tensors="pt",
                                                   padding=True, truncation=True, max_length=1024)

                batch["target_text"] = [str(sample["answer"])]
                batch["target"] = t5model.tokenizer(batch["target_text"],
                                                    return_tensors="pt",
                                                    padding=True, truncation=True, max_length=1024)

                pred_texts, pred_tensors = t5model.forward_batch_eval(batch)

                input_text_all.extend(batch["input_text"])
                pred_text_all.extend(pred_texts)
                target_text_all.extend(batch["target_text"])

                if debug_flag:
                    print("-" * 40)
                    print(input_text_all[-1])
                    print(pred_text_all[-1])
                    print(target_text_all[-1])
                    input("-" * 40)

            hit_list = [
                ExpMetricUtils.get_seq2seq_em(pred_text_all[i], target_text_all[i], data_pattern=_DATA_PATTERN.value)
                for i in range(len(target_text_all))
            ]
            eval_acc = sum(hit_list) / len(hit_list)

            print("depth:", depth, " eval acc:", eval_acc, " avg ctx len:", np.mean(context_length))

            file_for_error_analysis[depth] = {
                "instances": instances_eval,
                "pred_text": pred_text_all,
                "target_text": target_text_all
            }


def main(unused_argv):
    EvalDebugT5E2E.load_and_eval(debug_flag=_DEBUG_FLAG.value)


if __name__ == '__main__':
    app.run(main)
