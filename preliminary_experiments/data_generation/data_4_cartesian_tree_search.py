import json
import os
import random
from transformers import T5Tokenizer

from preliminary_experiments.data_generation.data_utils import DataUtils, DatasetUtils
from preliminary_experiments.data_generation.data_base_class import DataBase

from preliminary_experiments.data_generation.data_1_cartesian import GenerateCartesianData
from preliminary_experiments.data_generation.data_2_tree_search import GenerateTreeSearchData


class GenerateCartesianTreeSearchData(DataBase):

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    @classmethod
    def generate_id_from_context_using_hash(cls, context_string):
        cls.hash_module.update(context_string.encode("utf-8"))

        return cls.hash_module.hexdigest()

    @classmethod
    def generate_one_example(cls, depth, k=3, debug_flag=False):
        """First generate one cartesian example, then generate the tree search example on top of that.

        First generate one cartesian example, which has the fields:
         - id, depth,
         - context_string: each of ... has xxx, in natural language
         - question_string: list ..., in natural language.
         - answer: xxx has yyy, xxx has yyy, ...
         - target_list: the list of grounded (main_chara, quantity, item)
         - target_nl_list,
         - ungrounded_list, the list of ungrounded (main_chara, quantity, item)

        TODO: which k should we use?
        """

        # First generate one cartesian example:
        cartesian_depth = random.choice([2, 3]) if depth <= 2 else 4
        cartesian_instance = GenerateCartesianData.generate_one_example(depth=cartesian_depth)

        initial_s_grounded = cartesian_instance["target_list"]
        initial_s_ungrounded = cartesian_instance["ungrounded_list"]

        existing_grounded_chara_item = set([(s[0], tuple([2])) for s in initial_s_grounded])
        existing_grounded_chara_quant_item = set([(s[0], s[1], tuple(s[2])) for s in initial_s_grounded])
        existing_ungrounded_chara_quant_item = set([(s[0], s[1], tuple(s[2])) for s in initial_s_ungrounded])

        # The initial statements will be further sampled in the tree search example generation function
        tree_search_instance = GenerateTreeSearchData.generate_one_example(
            depth=depth, k=k, tokenizer=cls.tokenizer,
            initial_statements_grounded=initial_s_grounded,
            initial_statements_ungrounded=initial_s_ungrounded,
            existing_grounded_chara_item=existing_grounded_chara_item,
            existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item,
            names_to_subtract=set([])
        )

        # Construct the actual cartesian tree search example.
        cartesian_tree_search_instance = {
            "cartesian_instance": cartesian_instance,
            "tree_search_instance": tree_search_instance
        }

        if debug_flag:
            print("=" * 40)
            print(json.dumps(cartesian_instance, indent=2))
            print("-" * 40)
            print(json.dumps(tree_search_instance, indent=2))
            input("-" * 40)

        # What should be included in the final example:
        # the cartesian instance, the tree search instance, the textual input and the textual output
        # The textual output should be both w/ intermediate annotations and w/o/ intermediate annotations.

        num_sampled_statements = len(tree_search_instance["statement_indices_shuffle_map"])
        # This list has all components that needs to included in the input.
        textual_input_list = [
            cartesian_instance["context_string"],
            " ".join(tree_search_instance["context_list"][num_sampled_statements:]),
        ]
        textual_input = " ".join(textual_input_list)

        textual_target = tree_search_instance["answer"]

        textual_target_w_inter_list = [
            " ".join([f"{t_}." for t_ in cartesian_instance["target_nl_list"]]),
            tree_search_instance["target_text_w_inter"]
        ]

        textual_target_w_inter = " ".join(textual_target_w_inter_list)

        cartesian_tree_search_instance["answer"] = tree_search_instance["answer"]
        cartesian_tree_search_instance["depth"] = depth
        cartesian_tree_search_instance["context_string"] = textual_input
        cartesian_tree_search_instance["question_string"] = tree_search_instance["question_string"]
        cartesian_tree_search_instance["target_text"] = textual_target
        cartesian_tree_search_instance["target_text_w_inter"] = textual_target_w_inter

        if debug_flag:
            print("\n".join(textual_input_list))
            print("-" * 40)
            print("\n".join(textual_target_w_inter_list))
            input("---")

        GenerateCartesianTreeSearchDataRuntimeChecks.runtime_checks_all(cartesian_tree_search_instance)

        return cartesian_tree_search_instance

    @classmethod
    def generate_data_with_certain_depth(cls, depth, num_train, num_dev, num_test, debug_flag=False):

        random.seed(depth)

        splits = ["train", "dev", "test"]

        num_instances = {
            "train": num_train,
            "dev": num_dev,
            "test": num_test
        }

        instances_all_splits = {split: [] for split in splits}

        existing_instances = {}  # This is used to make sure we don't generate repeating examples.

        for split in splits:

            while len(instances_all_splits[split]) < num_instances[split]:

                instance = cls.generate_one_example(depth=depth, debug_flag=debug_flag)

                if instance["context_string"] not in existing_instances:
                    instances_all_splits[split].append(instance)
                    existing_instances[instance["context_string"]] = 1

                    instance["id"] = cls.generate_id_from_context_using_hash(instance["context_string"])

                    len_tokenized_input = len(cls.tokenizer(instance["context_string"])["input_ids"])
                    instance["context_len"] = len_tokenized_input

        return instances_all_splits

    @classmethod
    def generate_data_all_depths(cls):
        n_train = 10000
        n_dev = 1000
        n_test = 1000

        data_folder_dir = os.path.join(cls.project_data_folder_path, "cartesian_tree_search_v1.0/")

        if not os.path.exists(data_folder_dir):
            os.mkdir(data_folder_dir)

        data_with_various_depth_raw = {}
        for d in [0, 1, 2, 3, 4]:
            print("=" * 40)
            print("generating data with depth ", d)
            chaining_data = cls.generate_data_with_certain_depth(depth=d,
                                                                 num_train=n_train,
                                                                 num_dev=n_dev,
                                                                 num_test=n_test)

            data_with_various_depth_raw[d] = chaining_data

        chaining_data_by_du = {}
        for du in [2, 4]:
            n_train_per_depth = int(n_train / (du + 1))
            n_dev_per_depth = int(n_dev / (du + 1))
            n_test_per_depth = int(n_test / (du + 1))

            chaining_data_by_du[du] = {"train": [], "dev": [], "test": []}
            for d in range(du + 1):
                chaining_data_by_du[du]["train"].extend(data_with_various_depth_raw[d]["train"][:n_train_per_depth])
                chaining_data_by_du[du]["dev"].extend(data_with_various_depth_raw[d]["dev"][:n_dev_per_depth])
                chaining_data_by_du[du]["test"].extend(data_with_various_depth_raw[d]["test"][:n_test_per_depth])

            chaining_data_by_du[du]["statistics"] = DatasetUtils.get_dataset_statistics(chaining_data_by_du[du])

            print("=" * 40)
            print("du ", du)
            print("statistics:")
            print(json.dumps(chaining_data_by_du[du]["statistics"], indent=2))

            with open(data_folder_dir + "cartesian_tree_search_data_du" + str(du) + ".json", "w") as handle:
                json.dump(chaining_data_by_du[du], handle)


class GenerateCartesianTreeSearchDataRuntimeChecks(GenerateCartesianTreeSearchData):

    @classmethod
    def runtime_check_grounded_ungrounded_statements_in_tree_search(cls, instance):
        """The initial grounded statements and ungrounded statements should come from the correct place."""

        cartesian_grounded_s = set(instance["cartesian_instance"]["target_list"])
        cartesian_ungrounded_s = set(instance["cartesian_instance"]["ungrounded_list"])

        tree_search_grounded_s = set(instance["tree_search_instance"]["statements"]["grounded"][0])
        tree_search_ungrounded_s = set(instance["tree_search_instance"]["statements"]["ungrounded"][0])

        assert tree_search_grounded_s.issubset(cartesian_grounded_s)
        assert tree_search_ungrounded_s.issubset(cartesian_ungrounded_s)

    @classmethod
    def runtime_check_initial_grounded_ungrounded_no_overlap(cls, instance):
        grounded = set(instance["cartesian_instance"]["target_list"])
        ungrounded = set(instance["cartesian_instance"]["ungrounded_list"])

        assert len(grounded.intersection(ungrounded)) == 0

    @classmethod
    def runtime_check_all_grounded_ungrounded_no_overlap(cls, instance):
        grounded = set(instance["cartesian_instance"]["target_list"])
        for s_grounded in instance["tree_search_instance"]["statements"]["grounded"]:
            grounded.update(set(s_grounded))

        ungrounded = set(instance["cartesian_instance"]["ungrounded_list"])
        for s_ungrounded in instance["tree_search_instance"]["statements"]["ungrounded"]:
            ungrounded.update(set(s_ungrounded))

        assert len(grounded.intersection(ungrounded)) == 0

    @classmethod
    def runtime_check_no_name_overlap(cls, instance):
        """The entities of the tree search rules should not overlap with the grounded or ungrounded statements."""

        grounded_s_names = set([s[0] for s in instance["cartesian_instance"]["target_list"]])
        ungrounded_s_names = set([s[0] for s in instance["cartesian_instance"]["ungrounded_list"]])
        initial_s_names = grounded_s_names.union(ungrounded_s_names)

        tree_search_instance = instance["tree_search_instance"]
        all_rules = [r for step_r in tree_search_instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in tree_search_instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in tree_search_instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in tree_search_instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in tree_search_instance["rules"]["backtracking"] if r != None]

        all_rules_names = set([r[1][0] for r in all_rules])

        assert len(initial_s_names.intersection(all_rules_names)) == 0

    @classmethod
    def runtime_check_final_answer(cls, instance):
        """If the answer is Yes, it should come from the last step of grounded statements."""
        if instance["tree_search_instance"]["answer"] == "Yes":
            assert (instance["tree_search_instance"]["question"] in
                    instance["tree_search_instance"]["statements"]["grounded"][-1])
        else:
            assert (instance["tree_search_instance"]["question"] in
                    instance["tree_search_instance"]["statements"]["ungrounded"][-1])

    @classmethod
    def runtime_checks_all(cls, instance):
        cls.runtime_check_grounded_ungrounded_statements_in_tree_search(instance)
        cls.runtime_check_initial_grounded_ungrounded_no_overlap(instance)
        cls.runtime_check_all_grounded_ungrounded_no_overlap(instance)
        # cls.runtime_check_no_name_overlap(instance)
        cls.runtime_check_final_answer(instance)


if __name__ == "__main__":
    GenerateCartesianTreeSearchData.generate_data_all_depths()
