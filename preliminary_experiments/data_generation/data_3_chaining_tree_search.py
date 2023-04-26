import json
import numpy as np
import random
import hashlib
import os

from transformers import T5Tokenizer

from preliminary_experiments.data_generation.data_utils import DataUtils, DatasetUtils
from preliminary_experiments.data_generation.data_base_class import DataBase


from preliminary_experiments.data_generation.data_0_chaining import GenerateChainingData
from preliminary_experiments.data_generation.data_2_tree_search import GenerateTreeSearchData


class GenerateChainingTreeSearchData(DataBase):

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    @classmethod
    def generate_id_from_context_using_hash(cls, context_string):
        cls.hash_module.update(context_string.encode("utf-8"))

        return cls.hash_module.hexdigest()

    @classmethod
    def formal_statement_to_nl(cls, formal_statement):

        if int(formal_statement[1]) == 1:
            return f"{formal_statement[0]} had 1 {formal_statement[2][0]}."
        else:
            return f"{formal_statement[0]} had {formal_statement[1]} {formal_statement[2][1]}."

    @classmethod
    def generate_final_grounded_statements_from_chains(cls, chains):
        """
        For the generation of tree search data, we need to generate the final statements from the chain. The input
        should have the format: [(name, quantity, item)] * k.

        :param chains: a list of chains of a chaining example.
        :return:
        """

        final_statements = []
        final_statements_nl = []
        for chain in chains:
            main_role_name = chain["formal_reps"][0][0]
            quantity = chain["answer"]
            item = chain["formal_reps"][0][2]

            final_statements.append((main_role_name, quantity, item))

        for f_s in final_statements:
            final_statements_nl.append(cls.formal_statement_to_nl(f_s))

        return final_statements, final_statements_nl

    @classmethod
    def generate_final_ungrounded_statements_from_chains(cls, chains):
        """
        First generate the final statements from the chain, then do some perturbation to the numbers of the chains, so
        that they can be used for the ungrounded statements in the tree search data generation.

        Due to the way the chains are generated, it is already guaranteed that the (main, item) of the grounded and
        ungrounded chains will not overlap.
        :param chains: candidate chains for the ungrounded statements generation.
        :return:
        """

        # In the tree search data, it is guaranteed that
        candidate_quantity_list = range(cls.num_item_bound[0], cls.num_item_bound[1])

        final_statements = []
        final_statements_nl = []
        for chain in chains:
            main_role_name = chain["formal_reps"][0][0]
            old_quantity = chain["answer"]
            item = chain["formal_reps"][0][2]

            new_quantity = random.choice(candidate_quantity_list)
            while new_quantity == old_quantity:
                new_quantity = random.choice(candidate_quantity_list)

            final_statements.append((main_role_name, new_quantity, item))

        for f_s in final_statements:
            final_statements_nl.append(cls.formal_statement_to_nl(f_s))

        return final_statements, final_statements_nl

    @classmethod
    def generate_final_statements_all_chains(cls, chains):
        """Generate the final statements from all chains. These will be used for the supervision data."""
        final_statements = []
        final_statements_nl = []
        for chain in chains:
            main_role_name = chain["formal_reps"][0][0]
            old_quantity = chain["answer"]
            item = chain["formal_reps"][0][2]

            final_statements.append((main_role_name, old_quantity, item))

        for f_s in final_statements:
            final_statements_nl.append(cls.formal_statement_to_nl(f_s))

        return final_statements, final_statements_nl

    @classmethod
    def generate_names_to_avoid_for_tree_search_from_chaining(cls, chains):

        names_to_subtract = set([])

        for chain in chains:
            for step_idx in range(1, len(chain["formal_reps"])):
                role_name = chain["formal_reps"][step_idx][0]

                names_to_subtract.add(role_name)

        return names_to_subtract

    @classmethod
    def generate_one_example(cls, depth, k=3,
                             debug_flag=False, debug_chaining_flag=False, debug_tree_search_flag=False):
        """
        Generate 1 example by first generate the chaining part then the tree search part

        For the tree search data, there are 3 to 5 initial grounded statements. So when generating the chaining data,
            we need to generate at least (3~5) * depth statements. That is, for the depth 5 data, we might need 25
            statements in the chaining part (excluding the distractors of chaining data)

        The working flow of this function:
         - (1) determine how many initial statements we want to have.
         - (2) sample the main characters of the chaining data given the sampled number
         - (3) generate the chaining data for each sampled main character
         - (4) given the generated chaining statements, generate the tree search data.
         - (5) check the generated structured data.
         - (6) translate them to natural language
        :return:
        """

        n_initial_statements_grounded = random.randint(2, k)  # Both number included.
        n_initial_statements_ungrounded = n_initial_statements_grounded

        # Here we first generate some chaining instances for the tree search data generation.
        # Some statements will be used as the grounded statements of the tree-search data, others will be
        # used as the ungrounded statements.

        # First generate some chaining instances.
        chaining_instance = GenerateChainingData.generate_one_example(
            depth=depth, num_chains=n_initial_statements_grounded + n_initial_statements_ungrounded,
            initial_statements=set([]), existing_grounded_chara_item=set([]),
            existing_grounded_chara_quant_item=set([]), existing_ungrounded_chara_quant_item=set([])
        )
        chaining_instance_chains = chaining_instance["chains"]
        chains_grounded = random.sample(chaining_instance_chains, n_initial_statements_grounded)
        chains_ungrounded = [c for c in chaining_instance_chains if c not in chains_grounded]

        (initial_s_grounded,
         initial_s_grounded_nl) = cls.generate_final_grounded_statements_from_chains(
            chains_grounded)
        (initial_s_ungrounded,
         initial_s_ungrounded_ungrounded_nl) = cls.generate_final_ungrounded_statements_from_chains(
            chains_ungrounded)
        (initial_s_all,
         initial_s_all_nl) = cls.generate_final_statements_all_chains(chaining_instance_chains)

        existing_grounded_chara_item = set([(s[0], tuple([2])) for s in initial_s_grounded])
        existing_grounded_chara_quant_item = set([(s[0], s[1], tuple(s[2])) for s in initial_s_grounded])
        existing_ungrounded_chara_quant_item = set([(s[0], s[1], tuple(s[2])) for s in initial_s_ungrounded])

        names_to_subtract = cls.generate_names_to_avoid_for_tree_search_from_chaining(chaining_instance_chains)

        tree_search_instance = GenerateTreeSearchData.generate_one_example(
            depth=depth, k=k, tokenizer=cls.tokenizer,
            initial_statements_grounded=initial_s_grounded,
            initial_statements_ungrounded=initial_s_ungrounded,
            existing_grounded_chara_item=existing_grounded_chara_item,
            existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item,
            names_to_subtract=names_to_subtract
        )

        # Construct the actual cartesian tree search example.
        chaining_tree_search_instance = {
            "chaining_instance": chaining_instance,
            "tree_search_instance": tree_search_instance
        }

        if debug_flag:
            print("=" * 40)
            print(json.dumps(chaining_instance, indent=2))
            print("-" * 40)
            print(json.dumps(tree_search_instance, indent=2))
            input("-" * 40)

        num_sampled_statements = len(tree_search_instance["statement_indices_shuffle_map"])
        # This list has all components that needs to included in the input.
        textual_input_list = [
            chaining_instance["context_string"],
            " ".join(tree_search_instance["context_list"][num_sampled_statements:]),
        ]
        textual_input = " ".join(textual_input_list)

        textual_target = tree_search_instance["answer"]

        textual_target_w_inter_list = [
            " ".join(initial_s_all_nl),
            tree_search_instance["target_text_w_inter"]
        ]

        textual_target_w_inter = " ".join(textual_target_w_inter_list)

        chaining_tree_search_instance["depth"] = depth
        chaining_tree_search_instance["initial_s_grounded"] = initial_s_all
        chaining_tree_search_instance["initial_s_grounded_selected"] = initial_s_grounded
        chaining_tree_search_instance["initial_s_ungrounded"] = initial_s_ungrounded
        chaining_tree_search_instance["context_string"] = textual_input
        chaining_tree_search_instance["question_string"] = tree_search_instance["question_string"]
        chaining_tree_search_instance["target_text"] = textual_target
        chaining_tree_search_instance["target_text_w_inter"] = textual_target_w_inter
        chaining_tree_search_instance["answer"] = tree_search_instance["answer"]

        if debug_flag:
            print("\n".join(textual_input_list))
            print("-" * 40)
            print("\n".join(textual_target_w_inter_list))
            input("---")

        GenerateChainingTreeSearchDataRuntimeChecks.runtime_checks_all(chaining_tree_search_instance)

        return chaining_tree_search_instance

    @classmethod
    def generate_data_with_certain_depth(cls, depth, k, num_train, num_dev, num_test, debug_flag=False):
        """
        Generate the data containing only a certain depth.
        :return:
        """

        random.seed(depth)

        splits = ["train", "dev", "test"]

        num_instances = {
            "train": num_train,
            "dev": num_dev,
            "test": num_test
        }

        instances_all_splits = {split: [] for split in splits}

        existing_instances = {}  # This is used to make sure we don't generate repeating examples.

        tokenizer = T5Tokenizer.from_pretrained("t5-small")  # This is used to check the length of the input

        for split in splits:

            while len(instances_all_splits[split]) < num_instances[split]:

                instance = cls.generate_one_example(depth=depth, k=k)

                if instance["context_string"] not in existing_instances:
                    instances_all_splits[split].append(instance)
                    existing_instances[instance["context_string"]] = 1

                    instance["id"] = cls.generate_id_from_context_using_hash(instance["context_string"])

                    len_tokenized_input = len(tokenizer(instance["context_string"], truncation=False)["input_ids"])
                    instance["context_len"] = len_tokenized_input

        return instances_all_splits

    @classmethod
    def generate_data_all_depths(cls):
        n_train = 10000
        n_dev = 1000
        n_test = 1000

        data_folder_dir = os.path.join(cls.project_data_folder_path, "chaining_tree_search_v1.0/")

        if not os.path.exists(data_folder_dir):
            os.mkdir(data_folder_dir)

        data_with_various_depth_raw = {}
        for d in [0, 1, 2, 3, 4]:
            print("=" * 40)
            print("generating data with depth ", d)
            chaining_data = cls.generate_data_with_certain_depth(depth=d,
                                                                 k=3,
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

            with open(data_folder_dir + "chaining_tree_search_data_du" + str(du) + ".json", "w") as handle:
                json.dump(chaining_data_by_du[du], handle)


class GenerateChainingTreeSearchDataRuntimeChecks(GenerateChainingTreeSearchData):

    @classmethod
    def runtime_check_grounded_ungrounded_statements_in_tree_search(cls, instance):
        """The initial grounded statements and ungrounded statements should come from the correct place."""

        chaining_grounded_s = set(instance["initial_s_grounded"])
        chaining_ungrounded_s = set(instance["initial_s_ungrounded"])

        tree_search_grounded_s = set(instance["tree_search_instance"]["statements"]["grounded"][0])
        tree_search_ungrounded_s = set(instance["tree_search_instance"]["statements"]["ungrounded"][0])

        assert tree_search_grounded_s.issubset(chaining_grounded_s)
        assert tree_search_ungrounded_s.issubset(chaining_ungrounded_s)

    @classmethod
    def runtime_check_initial_grounded_ungrounded_no_overlap(cls, instance):
        grounded = set(instance["initial_s_grounded"])
        ungrounded = set(instance["initial_s_ungrounded"])

        assert len(grounded.intersection(ungrounded)) == 0

    @classmethod
    def runtime_check_all_grounded_ungrounded_no_overlap(cls, instance):
        grounded = set(instance["initial_s_grounded"])
        for s_grounded in instance["tree_search_instance"]["statements"]["grounded"]:
            grounded.update(set(s_grounded))

        ungrounded = set(instance["initial_s_ungrounded"])
        for s_ungrounded in instance["tree_search_instance"]["statements"]["ungrounded"]:
            ungrounded.update(set(s_ungrounded))

        if len(grounded.intersection(ungrounded)) != 0:
            print("=" * 40)
            print(instance["initial_s_grounded"])
            print(instance["tree_search_instance"]["statements"]["grounded"])
            print("-" * 40)
            print(instance["initial_s_ungrounded"])
            print(instance["tree_search_instance"]["statements"]["ungrounded"])
            input("----")

        assert len(grounded.intersection(ungrounded)) == 0

    @classmethod
    def runtime_check_no_name_overlap(cls, instance):
        """The entities of the tree search rules should not overlap with the grounded or ungrounded statements."""

        grounded_s_names = set([s[0] for s in instance["initial_s_grounded"]])
        ungrounded_s_names = set([s[0] for s in instance["initial_s_ungrounded"]])
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
        cls.runtime_check_initial_grounded_ungrounded_no_overlap(instance)
        cls.runtime_check_all_grounded_ungrounded_no_overlap(instance)
        # cls.runtime_check_no_name_overlap(instance)
        cls.runtime_check_final_answer(instance)
        cls.runtime_check_grounded_ungrounded_statements_in_tree_search(instance)


if __name__ == "__main__":
    GenerateChainingTreeSearchData.generate_data_all_depths()
