import json
import random
import re

from preliminary_experiments.data_generation.data_utils import DataUtils


class ExpDatasetUtils:

    data_folder_paths = {
        "mac": "/Users/curry/zhengzhong/research/2022_NLTuringMachine/data/",
        "alix": "/home/zhengzhongliang/CLU_Projects/2022_IntermediateAnnotation/data/",
        "hpc": "/home/u15/zhengzhongliang/2022_IntermediateAnnotation/data/"
    }

    @classmethod
    def remove_dataset_version(cls, dataset_name_with_version):

        match_pattern = r'_v\d+.\d+'

        dataset_name_without_version = re.sub(match_pattern, "", dataset_name_with_version)

        return dataset_name_without_version

    @classmethod
    def parse_tasks(cls, tasks):

        tasks_w_version = [t for t in tasks.split("-") if t != ""]

        return [cls.remove_dataset_version(t) for t in tasks_w_version]

    @classmethod
    def load_data(cls, seed=None, n_train=20000, machine_switch="alix", data_pattern="chaining", dev_ratio=0.1):
        """
        This function loads the data and also sample the data if specified. This function should be used in all models
            and all experiments to load the data.

        :param seed: the seed used to sample the data
        :param n_train: how many training samples to use
        :param machine_switch: what machine this script is run on
        :param data_pattern: chaining, cartesian, tree search, chaining + tree search, cartesian + tree search, and all
        :param dev_ratio: the ratio of num_dev / num_train.
        :return:
        """

        data_patterns = ["chaining", "cartesian", "tree_search",
                         "chaining_tree_search", "cartesian_tree_search", "chaining_cartesian_tree_search"]

        data_pattern_no_version = cls.remove_dataset_version(data_pattern)
        assert data_pattern_no_version in data_patterns

        data_folder_path = cls.data_folder_paths[machine_switch]
        data_path = {
            "chaining": {
                2: data_folder_path + data_pattern + "/chaining_data_du2.json",
                4: data_folder_path + data_pattern + "/chaining_data_du4.json",
            },
            "cartesian": {
                3: data_folder_path + data_pattern + "/cartesian_data_du3.json",
                4: data_folder_path + data_pattern + "/cartesian_data_du4.json",
            },
            "tree_search": {
                2: data_folder_path + data_pattern + "/tree_search_data_du2.json",
                4: data_folder_path + data_pattern + "/tree_search_data_du4.json",
            },
            "chaining_tree_search": {
                2: data_folder_path + data_pattern + "/chaining_tree_search_data_du2.json",
                4: data_folder_path + data_pattern + "/chaining_tree_search_data_du4.json",
            },
            "cartesian_tree_search": {
                2: data_folder_path + data_pattern + "/cartesian_tree_search_data_du2.json",
                4: data_folder_path + data_pattern + "/cartesian_tree_search_data_du4.json",
            },
        }[data_pattern_no_version]

        if data_pattern_no_version == "cartesian":
            instances_all_du = {
                3: DataUtils.load_json(data_path[3]),
                4: DataUtils.load_json(data_path[4]),
            }
        else:
            instances_all_du = {
                2: DataUtils.load_json(data_path[2]),
                4: DataUtils.load_json(data_path[4]),
            }

        n_train_each_du = {
            du: min(n_train, len(instances_all_du[du]["train"])) for du in instances_all_du.keys()
        }
        n_dev_each_du = {
            du: min(int(n_train * dev_ratio), len(instances_all_du[du]["dev"])) for du in instances_all_du.keys()
        }

        random.seed(seed)  # This is necessary to ensure reproducibility
        for depth in instances_all_du.keys():
            instances_all_du[depth]["train"] = random.sample(instances_all_du[depth]["train"],
                                                             n_train_each_du[depth])
            instances_all_du[depth]["dev"] = random.sample(instances_all_du[depth]["dev"],
                                                           n_dev_each_du[depth])

        train_du = 2 if 2 in instances_all_du else 3
        print("*" * 40)
        print("data loaded from ", data_path)
        print(f"training du: {train_du}, training data statistics:")
        print("n train: ", len(instances_all_du[train_du]["train"]),
              " n dev:", len(instances_all_du[train_du]["dev"]),
              " n test:", len(instances_all_du[train_du]["test"]))
        print("*" * 40)

        return instances_all_du

    @classmethod
    def load_data_evr(cls, seed=None, n_train=20000, machine_switch="alix", data_pattern="chaining", dev_ratio=0.1):
        """
        This function loads the data and also sample the data if specified. This function should be used in all models
            and all experiments to load the data.

        :param seed: the seed used to sample the data
        :param n_train: how many training samples to use
        :param machine_switch: what machine this script is run on
        :param data_pattern: chaining, cartesian, tree search, chaining + tree search, cartesian + tree search, and all
        :param dev_ratio: the ratio of num_dev / num_train.
        :return:
        """

        data_patterns = ["chaining_v0.4_evr_v0.1", "cartesian", "tree_search_v0.5",
                         "chaining_tree_search", "cartesian_tree_search", "chaining_cartesian_tree_search"]
        assert data_pattern in data_patterns

        data_folder_path = cls.data_folder_paths[machine_switch]
        data_path = {
            "chaining_v0.4_evr_v0.1": {
                2: data_folder_path + data_pattern + "/" + data_pattern + "_du2.json",
                5: data_folder_path + data_pattern + "/" + data_pattern + "_du5.json",
            },
            "cartesian": {
                2: data_folder_path + "",
                5: data_folder_path + "",
            },
            "tree_search": {
                2: data_folder_path + "",
                5: data_folder_path + "",
            },
            "chaining_tree_search": {
                2: data_folder_path + "",
                5: data_folder_path + "",
            },
            "cartesian_tree_search": {
                2: data_folder_path + "",
                5: data_folder_path + "",
            },
            "chaining_cartesian_tree_search": {
                2: data_folder_path + "",
                5: data_folder_path + "",
            },
        }[data_pattern]

        # Temporarily do this experiment: train on DU2 and test on DU2 + DU5
        instances_all_du = {
            2: DataUtils.load_json(data_path[2]),
            5: DataUtils.load_json(data_path[5]),
        }

        n_train_each_du = {
            du: min(n_train, len(instances_all_du[du]["train"])) for du in instances_all_du.keys()
        }
        n_dev_each_du = {
            du: min(int(n_train * dev_ratio), len(instances_all_du[du]["dev"])) for du in instances_all_du.keys()
        }

        random.seed(seed)  # This is necessary to ensure reproducibility
        for depth in instances_all_du.keys():
            instances_all_du[depth]["train"] = random.sample(instances_all_du[depth]["train"],
                                                             n_train_each_du[depth])
            instances_all_du[depth]["dev"] = random.sample(instances_all_du[depth]["dev"],
                                                           n_dev_each_du[depth])

        print("*" * 40)
        print("data loaded from ", data_path, " before flatten:")
        print("n train: ", len(instances_all_du[2]["train"]),
              " n dev:", len(instances_all_du[2]["dev"]),
              " n test:", len(instances_all_du[2]["test"]))
        print("*" * 40)

        # We just return the not flattened data, so that later in the experiment we can flatten the data by the needs.

        return instances_all_du

    @classmethod
    def load_data_debug(cls):

        data = cls.load_data(
            seed=1, n_train=500, machine_switch="mac", data_pattern="chaining_tree_search_v0.3", dev_ratio=0.1)

        print(data[2]["train"][100])


if __name__ == "__main__":

    ExpDatasetUtils.load_data_debug()
