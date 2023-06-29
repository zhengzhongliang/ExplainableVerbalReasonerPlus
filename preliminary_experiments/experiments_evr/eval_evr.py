import re
import os
import numpy as np
import time
import json
import torch

from absl import app
from absl import flags

from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent
from preliminary_experiments.experiments_evr.evr_class.evr_backbone_lm import T5ModelEVR
from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils
from preliminary_experiments.experiments_evr.evr_dataset import TasksDataset


FLAGS = flags.FLAGS
_TASK_NAME = flags.DEFINE_string("task_name", "chaining_v0.1", "the task name, e.g., tree_search_v0.5")
_DU = flags.DEFINE_integer("du", 2, "the depth of the training data. DU2 means the data with depth up to 2.")
_N_TRAIN = flags.DEFINE_integer("n_train", 500, "number of training instances")
_MODEL_NAME = flags.DEFINE_string("model_name", "unifiedqa-t5-base", "the neural model's name")
_MODEL_SEED = flags.DEFINE_integer("model_seed", 0, "the random seed to use to train the model")
_EVAL_DEPTH = flags.DEFINE_integer("eval_depth", 0, "the depth of the evaluation data")
_MACHINE_SWITCH = flags.DEFINE_string("machine_switch", "hpc", "which machine to use")
_NEURAL_MODULE_LOAD_PATH = flags.DEFINE_string("neural_module_load_path", "TODO", "the path of a trained neural module")
_SAVE_FOLDER_PATH = flags.DEFINE_string("save_folder_path", "TODO", "the folder to save the result")

_EVAL_START = flags.DEFINE_integer("eval_start", 0, "The start index of all instances for evaluation")
_EVAL_END = flags.DEFINE_integer("eval_end", 200, "The end index of all instances for all evaluation")


class NeuralModule:

    def __init__(self,
                 t5model
                 ):
        self.t5model = t5model
        self.t5model.model.eval()

    def inference(self, textual_input):

        """
        Need to do a few pure rule based output to make sure the whole workflow is fine
        :param textual_input:
        :return:
        """

        pred_text = self.t5model.forward_text(textual_input)

        return pred_text


class EVREvaluatorFormal:

    def __init__(self,
                 neural_module_load_path,
                 task_name,
                 du,
                 save_folder_path,
                 model_seed,
                 machine_switch,
                 n_train,
                 model_name,
                 eval_depth,
                 print_every=5,
                 debug_flag=False
                 ):

        self.neural_module_load_path = neural_module_load_path
        self.task_name = task_name
        self.du = du
        self.eval_depth = eval_depth
        self.eval_start = _EVAL_START.value
        self.eval_end = _EVAL_END.value

        self.save_folder_path = save_folder_path
        self.result_file_name = (f"inf_result_{task_name}_{model_name}_n_train_{n_train}_train_du_{du}_"
                                 f"eval_depth_{eval_depth}_{self.eval_start}_{self.eval_end}.json")

        t5model = T5ModelEVR(
            device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
            model_load_path=neural_module_load_path,
        )

        self.neural_module = NeuralModule(t5model)

        self.evr_agent = EVRAgent(neural_module=self.neural_module, debug_flag=debug_flag)

        self.model_seed = model_seed
        self.machine_switch = machine_switch
        self.n_train = n_train
        self.print_every = print_every

        self.debug_flag = debug_flag

    @classmethod
    def get_cartesian_hit(cls, preds, answer):

        s_cleaned = []
        for s in preds:
            if s.startswith('\'') and s.endswith('\''):
                s = s[1: -1]
            if s.endswith("."):
                s = s[: -1]
            s_cleaned.append(s)

        answers = answer.split(", ")
        answers = [a[:-1] if a.endswith(".") else a for a in answers]

        if set(s_cleaned) == set(answers):
            return 1
        else:
            return 0

    @classmethod
    def get_evr_hit(cls, episodic_buffer, answer, task_name):

        if "chaining_v" in task_name:

            if "episodic_buffer_4" in episodic_buffer:
                pred_ = re.findall(r"(\d+)", episodic_buffer["episodic_buffer_4"])
                if len(pred_) > 0 and pred_[0] == str(answer):
                    return 1
                else:
                    return 0
            else:
                return 0

        elif "cartesian_v" in task_name:

            if "episodic_buffer_3" in episodic_buffer:
                pred_ = episodic_buffer["episodic_buffer_3"]
                hit = cls.get_cartesian_hit(pred_, answer)
                return hit
            else:
                return 0

        elif "cartesian_tree_search_v" in task_name:
            if "episodic_buffer_3" in episodic_buffer:
                pred = "Yes" if len(re.findall(r"Chunk \d+ can", episodic_buffer["episodic_buffer_3"])) > 0 else "No"
                hit = 1 if pred == answer else 0
                return hit
            else:
                return 0

        elif "chaining_tree_search_v" in task_name:
            if "episodic_buffer_4" in episodic_buffer:
                pred = "Yes" if len(re.findall(r"Chunk \d+ can", episodic_buffer["episodic_buffer_4"])) > 0 else "No"
                hit = 1 if pred == answer else 0
                return hit
            else:
                return 0

        elif "tree_search_v" in task_name:
            if "episodic_buffer_2" in episodic_buffer:
                pred = "Yes" if len(re.findall(r"Chunk \d+ can", episodic_buffer["episodic_buffer_2"])) > 0 else "No"
                hit = 1 if pred == answer else 0
                return hit
            else:
                return 0

        else:
            return 0

    def evaluate_evr(self, debug_flag=False):

        instances_all_du = ExpDatasetUtils.load_data(seed=self.model_seed,
                                                     n_train=self.n_train,
                                                     data_pattern=self.task_name,
                                                     dev_ratio=0.1)

        instances_eval_ = TasksDataset.get_evr_eval_instances(instances_all_du, _TASK_NAME.value)[self.du]["test"]
        instances_eval = [in_ for in_ in instances_eval_ if in_["depth"] == _EVAL_DEPTH.value]

        print("*" * 40)
        print(f"Num total test examples: {len(instances_eval_)}")
        print(f"Num test examples current depth: {len(instances_eval)}")
        del instances_eval_
        print("*" * 40)

        hit_list = []
        id_list = []
        idx_list = []
        result_save_path = os.path.join(self.save_folder_path, self.result_file_name)
        for inst_idx, instance in enumerate(instances_eval[self.eval_start: self.eval_end]):

            inf_time = 0
            episodic_buffer_dict = instance["episodic_buffer_dict"]
            external_chunk = instance["external_chunks"]

            if debug_flag:
                print("=" * 40)
                print(json.dumps(external_chunk, indent=2))
                print("-" * 40)
                print(json.dumps(episodic_buffer_dict, indent=2))

            try:
                all_episodic_buffer_keys = list(episodic_buffer_dict.keys())
                all_episodic_buffer_keys = ", ".join(all_episodic_buffer_keys)

                start_time = time.time()
                _, _, episodic_buffer_returned, external_chunk_returned = self.evr_agent.new_mem_handler(
                    program_lines_parent_level=[f"new_mem({all_episodic_buffer_keys})"],
                    program_counter_parent_level=0,
                    local_variable_dict_parent_level={},
                    episodic_buffer_dict_parent_level=episodic_buffer_dict,
                    external_textual_buffer_dict=external_chunk
                )
                end_time = time.time()
                inf_time = end_time - start_time

            except KeyboardInterrupt:
                break

            except:
                episodic_buffer_returned = {}

            hit_flag = self.get_evr_hit(episodic_buffer_returned,
                                        instance["answer"],
                                        task_name=self.task_name)
            hit_list.append(hit_flag)
            id_list.append(instance["id"])
            idx_list.append(inst_idx + self.eval_start)

            if debug_flag:
                print("-" * 40)
                print(episodic_buffer_returned)
                print(f"instance depth: {instance['depth']}")
                print(f"answer: {instance['answer']}")
                print(f"hit? {hit_flag}")
                print(f"inference time: {inf_time}s")
                input("-" * 40)

            if (inst_idx + 1) % self.print_every == 0:
                print(f"evaluating instance {inst_idx}, acc: {np.mean(hit_list)}, inf time: {inf_time}")
                with open(result_save_path, "w") as handle:
                    json.dump({"id_list": id_list, "hit_list": hit_list, "idx_list": idx_list}, handle)


def main(unused_argv):
    print("*" * 40)
    print("DU:", _DU.value, " n train:", _N_TRAIN.value)
    print("seed:", _MODEL_SEED.value, " data pattern:", _TASK_NAME.value)
    print("eval depth:", _EVAL_DEPTH.value)
    print("Cuda available:", torch.cuda.is_available())
    print("*" * 40)

    evr_evaluator_formal = EVREvaluatorFormal(
        neural_module_load_path=_NEURAL_MODULE_LOAD_PATH.value,
        task_name=_TASK_NAME.value,
        du=_DU.value,
        save_folder_path=_SAVE_FOLDER_PATH.value,
        model_seed=_MODEL_SEED.value,
        machine_switch=_MACHINE_SWITCH.value,
        n_train=_N_TRAIN.value,
        model_name=_MODEL_NAME.value,
        eval_depth=_EVAL_DEPTH.value,
        print_every=1,
        debug_flag=False
    )
    evr_evaluator_formal.evaluate_evr(debug_flag=False)


if __name__ == "__main__":
    app.run(main)
