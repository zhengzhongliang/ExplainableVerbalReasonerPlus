from torch.utils.data import Dataset

from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils
from preliminary_experiments.data_generation.data_0_chaining_evr import GenerateEVRChainingData
from preliminary_experiments.data_generation.data_1_cartesian_evr import GenerateEVRCartesianData
from preliminary_experiments.data_generation.data_2_tree_search_evr import GenerateEVRTreeSearchData
from preliminary_experiments.data_generation.data_3_chaining_tree_search_evr import GenerateEVRChainingTreeSearchData
from preliminary_experiments.data_generation.data_4_cartesian_tree_search_evr import GenerateEVRCartesianTreeSearchData


class PadCollate:

    def __init__(self,
                 tokenizer,
                 max_len=1024,
                 model_name="allenai/unifiedqa-t5-base"):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

    def pad_collate(self, batch):

        batch_to_return = {}

        batch_to_return["input_text"] = [sample["input"] for sample in batch]
        batch_to_return["input"] = self.tokenizer(batch_to_return["input_text"],
                                                  return_tensors="pt",
                                                  padding=True, truncation=True, max_length=self.max_len)

        batch_to_return["target_text"] = [sample["target"] for sample in batch]
        batch_to_return["target"] = self.tokenizer(batch_to_return["target_text"],
                                                   return_tensors="pt",
                                                   padding=True, truncation=True, max_length=self.max_len)

        # Set the padding tokens to -100
        batch_to_return["target"]['input_ids'][
            batch_to_return["target"]['input_ids'] == self.tokenizer.pad_token_id] = -100

        # TODO: change this to a more elegant way later
        batch_to_return["id"] = [0 for sample in batch]

        batch_to_return["task_pattern"] = [str(sample["task"]) + "-" + str(sample["pattern"]) for sample in batch]

        return batch_to_return

    def __call__(self, batch):
        return self.pad_collate(batch)


class TasksDataset(Dataset):

    evr_gen_funcs = {
        "chaining": GenerateEVRChainingData.generate_evr_instances,
        "cartesian": GenerateEVRCartesianData.generate_evr_instances,
        "tree_search": GenerateEVRTreeSearchData.generate_evr_instances,
        "chaining_tree_search": GenerateEVRChainingTreeSearchData.generate_evr_instances,
        "cartesian_tree_search": GenerateEVRCartesianTreeSearchData.generate_evr_instances
    }

    evr_eval_gen_funcs = {
        "chaining": GenerateEVRChainingData.generate_evr_eval_instances,
        "cartesian": GenerateEVRCartesianData.generate_evr_eval_instances,
        "tree_search": GenerateEVRTreeSearchData.generate_evr_eval_instances,
        "chaining_tree_search": GenerateEVRChainingTreeSearchData.generate_evr_eval_instances,
        "cartesian_tree_search": GenerateEVRCartesianTreeSearchData.generate_evr_eval_instances
    }

    def __init__(self, instances):
        self.all_instances = instances

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        return self.all_instances[idx]

    @classmethod
    def get_evr_instances(cls, instances, task_name):

        task_name_no_ver = ExpDatasetUtils.remove_dataset_version(task_name)

        evr_instances = {}

        for du in instances.keys():
            if du not in evr_instances:
                evr_instances[du] = {}

            for split in ["train", "dev", "test"]:
                evr_instances[du][split] = cls.evr_gen_funcs[task_name_no_ver](instances[du][split])

        return evr_instances

    @classmethod
    def get_evr_eval_instances(cls, instances, task_name):

        task_name_no_ver = ExpDatasetUtils.remove_dataset_version(task_name)

        evr_instances = {}

        for du in instances.keys():
            if du not in evr_instances:
                evr_instances[du] = {}

            for split in ["train", "dev", "test"]:
                evr_instances[du][split] = cls.evr_eval_gen_funcs[task_name_no_ver](instances[du][split])

        return evr_instances
