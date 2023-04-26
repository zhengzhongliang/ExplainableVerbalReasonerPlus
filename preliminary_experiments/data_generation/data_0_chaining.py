import json
import numpy as np
import random
import os

from preliminary_experiments.data_generation.data_utils import DataUtils, DatasetUtils
from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils
from preliminary_experiments.data_generation.data_base_class import DataBase


class GenerateChainingData(DataBase):
    """This class generates the chaining data, with or without intermediate supervision."""

    @classmethod
    def generate_initial_statement_from_sampled_vars(cls, sampled_main_role_name, sampled_item, sampled_item_quantity):
        if sampled_item_quantity < cls.num_item_bound[0]:
            sampled_item_quantity = cls.num_item_bound[0]
        if sampled_item_quantity > cls.num_item_bound[1]:
            sampled_item_quantity = cls.num_item_bound[1]

        if sampled_item_quantity == 0:
            initial_statement = sampled_main_role_name + " had no " + sampled_item[0] + " in the beginning."
        elif sampled_item_quantity == 1:
            initial_statement = sampled_main_role_name + " had 1 " + sampled_item[0] + " in the beginning."
        else:
            initial_statement = (sampled_main_role_name + " had " + str(sampled_item_quantity) + " " +
                                 sampled_item[1] + " in the beginning.")

        return initial_statement

    @classmethod
    def generate_update_statement_from_sampled_vars(
            cls, sampled_main_role_name, sampled_other_role_name, sampled_item, sampled_op):

        # TODO: later we might want to add more variations to the dataset.
        op_num = int(sampled_op)
        noun_form_index = 0 if op_num in [-1, 1] else 1  # This determines whether to use single or plural form
        if op_num > 0:
            update_statement = (sampled_other_role_name + " gave " + sampled_main_role_name + " " +
                                str(op_num) + " " + sampled_item[noun_form_index] + ".")
        elif op_num == 0:
            update_statement = (sampled_other_role_name + " did not give " + sampled_main_role_name +
                                " any " + sampled_item[1] + ".")
        else:
            update_statement = (sampled_main_role_name + " gave " + sampled_other_role_name + " " +
                                str(abs(op_num)) + " " + sampled_item[noun_form_index] + ".")

        return update_statement

    @classmethod
    def generate_question_from_sampled_vars(cls, sampled_main_role_name, sampled_item):

        question = "How many " + sampled_item[1] + " did " + sampled_main_role_name + " have in the end?"

        return question

    @classmethod
    def generate_id_from_context_using_hash(cls, context_string):
        cls.hash_module.update(context_string.encode("utf-8"))

        return cls.hash_module.hexdigest()    # Return the hash as a string, in hex.

    @classmethod
    def generate_initial_statements(
        cls,
        initial_statements,
        n_statements_to_gen,
        existing_grounded_chara_item,
        existing_grounded_chara_quant_item,
        existing_ungrounded_chara_quant_item,
        debug_flag=False
    ):
        '''
        This function is copied from the tree search data generation code. Basically this can make the chaining
        data harder, and also easier when later when we want to merge it to the tree search data.
        '''

        if len(initial_statements) > 0:
            statements = initial_statements
            for statement_tuple in statements:
                existing_grounded_chara_item.add((statement_tuple[0], statement_tuple[2]))
                existing_grounded_chara_quant_item.add(statement_tuple)

        else:
            statements = []

            # A/B is the person name, X/Y is the item. AX is the original statement's person name and item name.
            statement_types = ["AmY", "AnY", "BmX", "BnX", "BmY", "BnY"]

            while len(statements) < n_statements_to_gen:

                # When there are more than 1 statement, give the option to choose harder conclusions.
                if len(statements) >= 1:
                    statement_type = random.choice(statement_types)
                    sampled_statement = random.choice(statements)
                    if statement_type[0] == "A":
                        main_chara = sampled_statement[0]
                    else:
                        main_chara = random.choice(cls.full_names)

                    if statement_type[1] == "m":
                        item_quantity = sampled_statement[1]
                    else:
                        item_quantity = random.randint(cls.num_item_bound[0], cls.num_item_bound[1])

                    if statement_type[2] == "X":
                        item = sampled_statement[2]
                    else:
                        item = random.choice(cls.items)

                    if debug_flag:
                        print("=" * 40)
                        print("existing statements:", statements)
                        print("sampled statement type:", statement_type)
                        print("sampled statement:", sampled_statement)

                else:
                    main_chara = random.choice(cls.full_names)  # choice: randomly select one element from the list
                    item_quantity = random.randint(cls.num_item_bound[0], cls.num_item_bound[1])
                    item = random.choice(cls.items)

                    if debug_flag:
                        print("=" * 40)
                        print("existing statements:", statements)

                statement_tuple = (main_chara, item_quantity, item)
                if ((main_chara, item) not in existing_grounded_chara_item and
                        statement_tuple not in existing_ungrounded_chara_quant_item):
                    statements.append(statement_tuple)
                    existing_grounded_chara_item.add((main_chara, item))
                    existing_grounded_chara_quant_item.add(statement_tuple)

                if debug_flag:
                    print("generated statement:", statement_tuple)
                    print("new statements:", statements)
                    print("grounded chara item:", existing_grounded_chara_item)
                    print("grounded chara quant item:", existing_grounded_chara_quant_item)
                    input("-" * 40)

        return (statements, existing_grounded_chara_item,
                existing_grounded_chara_quant_item, existing_ungrounded_chara_quant_item)

    @classmethod
    def generate_one_example(
        cls,
        depth,
        num_chains,
        initial_statements,
        existing_grounded_chara_item,
        existing_grounded_chara_quant_item,
        existing_ungrounded_chara_quant_item,
        debug_flag=False):
        # First sample a subject name and sample an item
        # If the main role is not none, it means it needs to use a already sampled main role
        if num_chains is None:
            num_chains = random.randint(3, 5)  # generate examples with 3 to 5 chains.

        chains = []
        (initial_statements, existing_grounded_chara_item, existing_grounded_chara_quant_item,
         existing_ungrounded_chara_quant_item) = cls.generate_initial_statements(
            initial_statements=initial_statements,
            n_statements_to_gen=num_chains, existing_grounded_chara_item=existing_grounded_chara_item,
            existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item)

        main_roles = [s[0] for s in initial_statements]
        for chain_idx in range(num_chains):
            formal_reps_one_chain = []  # Stores the formal representations of the sampled variables and operations
            quantity_ops_one_chain = []  # Stores the quantity at the beginning and the subsequent sampled operations
            instance_contexts_one_chain = []  # Stores the final generated statements as a list
            num_buffer_one_chain = 0  # Used for calculating the final answer

            main_role = initial_statements[chain_idx][0]
            item_quantity_beginning = initial_statements[chain_idx][1]
            main_item = initial_statements[chain_idx][2]  # This is a tuple, with both single and plural form

            initial_statement = cls.generate_initial_statement_from_sampled_vars(
                sampled_main_role_name=main_role, sampled_item=main_item, sampled_item_quantity=item_quantity_beginning)

            instance_contexts_one_chain.append(initial_statement)
            formal_reps_one_chain.append((main_role, item_quantity_beginning, main_item))
            quantity_ops_one_chain.append(item_quantity_beginning)
            num_buffer_one_chain = item_quantity_beginning

            # Then sample enough number of other names, sample time = depth, each time also sample an operation.

            # Sample other role names. If the sampled other role names include the main role name, remove it.
            sampled_other_role_names = []
            while len(sampled_other_role_names) < depth:
                sampled_other_role_name = random.choice(cls.full_names)
                if sampled_other_role_name not in main_roles:
                    sampled_other_role_names.append(sampled_other_role_name)

            for ctx_idx, other_role_name in enumerate(sampled_other_role_names):

                # Choose one operation from [-2, -1, 0, 1 ,2], and not exceeding 20 or less than 0
                sampled_operation = random.choice(cls.operations)
                while num_buffer_one_chain + int(sampled_operation) > cls.num_item_bound[1] or \
                        num_buffer_one_chain + int(sampled_operation) < cls.num_item_bound[0]:
                    sampled_operation = random.choice(cls.operations)

                update_context_statement = cls.generate_update_statement_from_sampled_vars(
                    sampled_main_role_name=main_role, sampled_other_role_name=other_role_name,
                    sampled_item=main_item, sampled_op=sampled_operation)

                formal_reps_one_chain.append((other_role_name, int(sampled_operation)))
                quantity_ops_one_chain.append(int(sampled_operation))
                instance_contexts_one_chain.append(update_context_statement)
                num_buffer_one_chain += int(sampled_operation)

            chains.append(
                {
                    "formal_reps": formal_reps_one_chain,
                    "quantity_ops": quantity_ops_one_chain,
                    "context_list": instance_contexts_one_chain,
                    "answer": num_buffer_one_chain
                }
            )

        # Randomly select one chain as the question
        selected_question_chain_idx = random.choice(range(num_chains))
        selected_chain = chains[selected_question_chain_idx]
        selected_question_answer = selected_chain["answer"]

        question_string = cls.generate_question_from_sampled_vars(
            sampled_main_role_name=selected_chain["formal_reps"][0][0],
            sampled_item=selected_chain["formal_reps"][0][2])

        concatenated_context = " ".join(" ".join(chain["context_list"]) for chain in chains)
        instance_id = cls.generate_id_from_context_using_hash(concatenated_context)

        instance = {
            "id": instance_id,
            "chains": chains,
            "selected_chain_idx": selected_question_chain_idx,
            "context_string": concatenated_context,
            "question_string": question_string,
            "answer": selected_question_answer,
            "depth": depth
        }

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

        # Stores the examples for all splits. Each example is distinguished by id.
        all_instances = {split: [] for split in splits}
        # This is used to keep track of what instances have been generated.
        instance_ids_all_splits = {}
        for split in splits:
            while len(all_instances[split]) < nums_per_split[split]:

                instance = cls.generate_one_example(
                    depth=depth, num_chains=None, initial_statements=[],
                    existing_grounded_chara_item=set(),
                    existing_grounded_chara_quant_item=set(),
                    existing_ungrounded_chara_quant_item=set())

                cls.runtime_checks_one_instance(instance)

                if instance["id"] not in instance_ids_all_splits:

                    all_instances[split].append(instance)
                    instance_ids_all_splits[instance["id"]] = 1

                    if debug_flag:
                        print("=" * 40)
                        print(json.dumps(instance, indent=2))
                        input("-" * 30)

        return all_instances

    @classmethod
    def _generate_inferred_statement_in_natural_language(cls, chara_name, quantity, item):
        """
        This function generates the inferred statement of each reasoning step for the all-at-once examples.
        :param chara_name:
        :param quantity:
        :param item:
        :return:
        """

        if quantity == 0:
            inferred_statement = chara_name + " had no " + item[0] + "."
        elif quantity == 1:
            inferred_statement = chara_name + " had 1 " + item[0] + "."
        else:
            inferred_statement = chara_name + " had " + str(quantity) + " " + item[1] + "."

        return inferred_statement

    @classmethod
    def generate_training_data_with_steps_all_at_once_one_instance(cls, instance):
        """
        Generate the training data for chaining with intermediate steps in the "at once" version.
        Only handle one instance.
        E.g., the target could be: A has X toys in the beginning; A give B Y toys, A have Z toys; .....
        :return:
        """

        selected_chain_idx = instance["selected_chain_idx"]
        selected_chain = instance["chains"][selected_chain_idx]

        target_chara = selected_chain["formal_reps"][0][0]
        buffer_quantity = selected_chain["formal_reps"][0][1]
        target_item = selected_chain["formal_reps"][0][2]

        target_list_first_statement = selected_chain["context_list"][0][:-1] + ";"  # change period to semi-colon

        statement_all_steps = [target_list_first_statement]
        # Form the target text of each step
        for step_idx in range(instance["depth"]):
            selected_evidence = selected_chain["context_list"][step_idx + 1]
            buffer_quantity += selected_chain["quantity_ops"][step_idx + 1]
            inferred_statement = cls._generate_inferred_statement_in_natural_language(target_chara,
                                                                                      buffer_quantity,
                                                                                      target_item)

            step_statement = selected_evidence[:-1] + ", " + inferred_statement[:-1] + ";"
            statement_all_steps.append(step_statement)

        # Finally, append a final answer to the generated statements:
        answer_statement = "Answer: " + str(buffer_quantity)
        statement_all_steps.append(answer_statement)

        assert (buffer_quantity) == int(instance["answer"])

        return statement_all_steps

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

        data_folder_dir = os.path.join(cls.project_data_folder_path, "chaining_v1.0/")

        if not os.path.exists(data_folder_dir):
            os.mkdir(data_folder_dir)

        data_with_various_depth_raw = {}
        for d in [0, 1, 2, 3, 4]:
            print("=" * 40)
            print(f"Generating chaining data depth {d}")
            chaining_data = cls.generate_data_with_certain_depth(depth=d,
                                                                 num_train=n_train,
                                                                 num_dev=n_dev,
                                                                 num_test=n_test)

            data_with_various_depth_raw[d] = chaining_data

        chaining_data_by_du = {}
        for du in [2, 4]:
            n_train_per_depth = int(n_train/(du + 1))
            n_dev_per_depth = int(n_dev/(du + 1))
            n_test_per_depth = int(n_test/(du + 1))

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

            with open(data_folder_dir + "chaining_data_du" + str(du) + ".json", "w") as handle:
                json.dump(chaining_data_by_du[du], handle)

    @classmethod
    def runtime_checks_one_instance(cls, instance):
        dataset_depth = instance["depth"]
        main_role_names = [chain["formal_reps"][0][0] for chain in instance["chains"]]

        assert (DataBase.num_item_bound[0] <= instance["answer"] <= DataBase.num_item_bound[
            1])  # The answer number should be between 0 and 20

        assert (sum(instance["chains"][instance["selected_chain_idx"]]["quantity_ops"]) == instance[
            "answer"])  # The sum of operations should equal to the answer

        for chain in instance["chains"]:
            assert (len(chain["context_list"]) == dataset_depth + 1)  # Check the reasoning depth of each example
            assert (DataBase.num_item_bound[0] <= chain["quantity_ops"][0] <= DataBase.num_item_bound[
                1])  # The beginning number should be between 0 and 20

            chain_other_names = [s[0] for s in chain["formal_reps"][1:]]

            assert (len(set(main_role_names).intersection(chain_other_names)) == 0)


if __name__ == "__main__":
    GenerateChainingData.generate_data_all_depth()
