import json
import numpy as np
import random
import hashlib
import os
import copy

from transformers import T5Tokenizer

from preliminary_experiments.data_generation.data_utils import DataUtils, DatasetUtils
from preliminary_experiments.data_generation.data_base_class import DataBase


class GenerateCartesianData(DataBase):

    tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")

    @classmethod
    def generate_id_from_context_using_hash(cls, context_string):
        cls.hash_module.update(context_string.encode("utf-8"))

        return cls.hash_module.hexdigest()  # Return the hash as a string, in hex.

    @classmethod
    def form_main_character_statement(cls, sampled_charas):

        if len(sampled_charas) == 1:
            statement = sampled_charas[0]
        else:
            statement = "Each of " + ", ".join(sampled_charas[:-1]) + " and " + sampled_charas[-1]

        return statement

    @classmethod
    def form_quantity_and_item_statement(cls, sampled_quantities, sampled_items):

        if len(sampled_quantities) == 1:
            statement = str(sampled_quantities[0]) + " " + sampled_items[0][0] \
                if sampled_quantities[0] == 1 else str(sampled_quantities[0]) + " " + sampled_items[0][1]

        else:
            quant_item_combined = [
                str(sampled_quantities[idx]) + " " + sampled_items[idx][0]
                if sampled_quantities[idx] == 1 else str(sampled_quantities[idx]) + " " + sampled_items[idx][1]
                for idx in range(len(sampled_quantities))
            ]

            statement = ", ".join(quant_item_combined[:-1]) + " and " + quant_item_combined[-1]

        return statement

    @classmethod
    def generate_positive_tuples(cls, sampled_charas, sampled_quantities, sampled_items):

        statement_tuples = []
        for chara_idx in range(len(sampled_charas)):
            for item_idx in range(len(sampled_items)):
                statement_tuples.append(
                    (sampled_charas[chara_idx], sampled_quantities[item_idx], sampled_items[item_idx])
                )

        return statement_tuples

    @classmethod
    def form_target(cls, sampled_charas, sampled_quantities, sampled_items):

        statement_tuples = cls.generate_positive_tuples(sampled_charas, sampled_quantities, sampled_items)

        target_statement_list = []
        target_statement_nl_list = []
        for statement_tuple in statement_tuples:
            one_statement_nl = statement_tuple[0] + " had " + str(statement_tuple[1]) + " " + \
                            (statement_tuple[2][0] if statement_tuple[1] == 1
                             else statement_tuple[2][1])

            target_statement_list.append(statement_tuple)
            target_statement_nl_list.append(one_statement_nl)

        target_statement_nl = ", ".join(target_statement_nl_list) + "."

        return target_statement_nl, target_statement_list, target_statement_nl_list

    @classmethod
    def sample_negative_examples(cls, grounded_statements, n_statements_to_gen, debug_flag=False):
        # This function is used to generate the examples for the discriminative models. The negative statements should
        # have the 4 times more statements than the positive examples.

        # Assuming the original grounded query is "AmX", A for chara name, m for quantity, X for item.
        statement_types = ["AmY", "AnX", "AnY", "BmX", "BmY", "BnX", "BnY"]

        ungrounded_statements = []
        existing_grounded_chara_quant_item = set(grounded_statements)
        existing_ungrounded_chara_quant_item = set([])

        while len(ungrounded_statements) < n_statements_to_gen:
            statement_type = random.choice(statement_types)
            sampled_grounded_statement = random.choice(grounded_statements)

            if debug_flag:
                print("=" * 40)
                print("grounded statements:", grounded_statements)
                print("sampled statement type:", statement_type)
                print("sampled grounded statement:", sampled_grounded_statement)

            if statement_type[0] == "A":
                main_chara = sampled_grounded_statement[0]
            else:  # distractor_type[0] == "B"
                main_chara = random.choice(cls.full_names)

            if statement_type[1] == "m":
                item_quantity = sampled_grounded_statement[1]
            else:
                item_quantity = random.randint(cls.num_item_bound[0], cls.num_item_bound[1])

            if statement_type[2] == "X":
                item = sampled_grounded_statement[2]
            else:
                item = random.choice(cls.items)

            statement_tuple = (main_chara, item_quantity, item)
            if statement_tuple not in existing_grounded_chara_quant_item and \
                    statement_tuple not in existing_ungrounded_chara_quant_item:
                ungrounded_statements.append(statement_tuple)
                existing_ungrounded_chara_quant_item.add(statement_tuple)

            if debug_flag:
                print("generated ungrounded statement:", statement_tuple)
                print("existing ungounded statements:", existing_ungrounded_chara_quant_item)
                input("-" * 40)

        ungrounded_statements_nl = [
            u_s_t[0] + " had " + str(u_s_t[1]) + " " + (u_s_t[2][0] if u_s_t[1] == 1 else u_s_t[2][1])
            for u_s_t in ungrounded_statements
        ]

        return ungrounded_statements, ungrounded_statements_nl

    @classmethod
    def check_grounded_statements(cls, instance):

        main_charas = []
        quant_items = []

        for statement_tuple in instance["target_list"]:
            main_charas.append(statement_tuple[0])
            quant_items.append((statement_tuple[1], statement_tuple[2]))

        assert len(set(main_charas)) == instance["depth"]
        assert len(set(quant_items)) == instance["depth"]

        assert len(set(instance["target_list"])) == instance["depth"] * instance["depth"]

        for main_chara in main_charas:
            assert main_chara in instance["context_string"]

        for quant_item in quant_items:
            assert str(quant_item[0]) + " " + quant_item[1][0] in instance["context_string"] or \
                str(quant_item[0]) + " " + quant_item[1][1] in instance["context_string"]

    @classmethod
    def check_ungrounded_statements(cls, instance):

        grounded_statement_sets = set(instance["target_list"])
        ungrounded_statement_sets = set(instance["ungrounded_list"])

        assert len(grounded_statement_sets.intersection(ungrounded_statement_sets)) == 0

    @classmethod
    def generate_one_example(cls, depth):

        # Sample names, the sampled names should not repeat
        sampled_names = random.sample(cls.full_names, depth)

        # Sample quantities, the sampled quantities can repeat
        sampled_quantities = []
        for d in range(depth):
            sampled_quantities.append(random.randint(cls.num_item_bound[0], cls.num_item_bound[1]))

        # Sample items, the sampled names should repeat
        sampled_items = random.sample(cls.items, depth)

        main_chara_statement = cls.form_main_character_statement(sampled_names)

        quant_item_statement = cls.form_quantity_and_item_statement(sampled_quantities, sampled_items)

        context = main_chara_statement + " had " + quant_item_statement + "."

        question = "List the items that each person had."

        target_statement_nl, target_statement_list, target_statement_nl_list = \
            cls.form_target(sampled_names, sampled_quantities, sampled_items)

        ungrounded_statement_list, ungrounded_statement_nl_list = \
            cls.sample_negative_examples(target_statement_list, len(target_statement_list))

        instance_id = cls.generate_id_from_context_using_hash(context + " " + question)

        len_tokenized_target = len(cls.tokenizer(target_statement_nl)["input_ids"])

        instance = {
            "id": instance_id,
            "depth": depth,
            "context_string": context,
            "question_string": question,
            "answer": target_statement_nl,
            "target_list": target_statement_list,
            "target_nl_list": target_statement_nl_list,
            "ungrounded_list": ungrounded_statement_list,
            "ungrounded_nl_list": ungrounded_statement_nl_list,
            "target_len": len_tokenized_target
        }

        cls.check_grounded_statements(instance)
        cls.check_ungrounded_statements(instance)

        return instance

    @classmethod
    def generate_data_with_certain_depth(cls, depth, num_train, num_dev, num_test, debug_flag=False):
        '''
        This function generates the chaining data with certain depth.
        Each instance is stored as json and should at least have the following fields:
         - ID, generated from md5 hash
         - structured input: ["name", "has", "num", "fruit_type"]
         - natural language expression
         - the correct answer.

        depth: number of steps of chaining.
        number of examples:
        :return:
        '''

        splits = ["train", "dev", "test"]
        nums_per_split = {"train": num_train, "dev": num_dev, "test": num_test}

        random.seed(depth)  # Use the number of depth as the random seed to ensure reproducibility

        all_instances = {split: [] for split in
                         splits}  # Stores the examples for all splits. Each example is distinguished by id.
        instance_ids_all_splits = {}  # This is used to keep track of what instances have been generated.
        for split in splits:
            while len(all_instances[split]) < nums_per_split[split]:

                instance = cls.generate_one_example(depth=depth)

                if instance["id"] not in instance_ids_all_splits:

                    all_instances[split].append(instance)
                    instance_ids_all_splits[instance["id"]] = 1

                    if debug_flag:
                        print("=" * 40)
                        print(json.dumps(instance, indent=2))
                        input("-" * 30)

        return all_instances

    @classmethod
    def generate_data_all_depth(cls):

        """
        This is for generating the debugging data to get the most basic understanding of the model's behaviors,
        including how many data are needed, what are the performance, what are the desired batch sizes and so on.

        Data statistics: only 10000 training samples, 2000 dev and 2000 test.

        The final data: DU2, DU3, DU4, DU5.
        :return:
        """

        n_train = 10000
        n_dev = 1000
        n_test = 1000

        data_folder_dir = os.path.join(cls.project_data_folder_path, "cartesian_v1.0/")

        os.makedirs(data_folder_dir, exist_ok=True)

        data_with_various_depth_raw = {}
        for d in [2, 3, 4]:   # Note that there is no meaning to generate depth 0 data for Cartesian product.
            print("=" * 40)
            print(f"Generating cartesian data depth {d}")
            cartesian_data = cls.generate_data_with_certain_depth(
                depth=d, num_train=n_train, num_dev=n_dev, num_test=n_test)

            data_with_various_depth_raw[d] = cartesian_data

        cartesian_data_by_du = {}
        for du in [3, 4]:
            n_train_per_depth = int(n_train / (du - 1))
            n_dev_per_depth = int(n_dev / (du - 1))
            n_test_per_depth = int(n_test / (du - 1))

            cartesian_data_by_du[du] = {"train": [], "dev": [], "test": []}
            for d in range(2, du + 1):
                cartesian_data_by_du[du]["train"].extend(data_with_various_depth_raw[d]["train"][:n_train_per_depth])
                cartesian_data_by_du[du]["dev"].extend(data_with_various_depth_raw[d]["dev"][:n_dev_per_depth])
                cartesian_data_by_du[du]["test"].extend(data_with_various_depth_raw[d]["test"][:n_test_per_depth])

                cartesian_data_by_du[du]["statistics"] = \
                    DatasetUtils.get_dataset_statistics_cartesian(cartesian_data_by_du[du])

            print("=" * 40)
            print("du ", du)
            print("statistics:")
            print(json.dumps(cartesian_data_by_du[du]["statistics"], indent=2))

            with open(data_folder_dir + "cartesian_data_du" + str(du) + ".json", "w") as handle:
                json.dump(cartesian_data_by_du[du], handle)


class CheckCartesian:

    @classmethod
    def check_cartesian_one_instance(cls):

        instances = DataUtils.load_json(
            "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/cartesian_v0.1/cartesian_data_du2.json"
        )

        for instance in instances["train"]:
            if instance["depth"] == 2:
                print(json.dumps(instance, indent=2))
                break


if __name__ == "__main__":
    #GenerateCartesianData.generate_data_with_certain_depth(5, 20000, 2000, 2000, debug_flag=False)
    GenerateCartesianData.generate_data_all_depth()
    #CheckCartesian.check_cartesian_one_instance()