import os
import json

import random

import numpy as np

from transformers import T5Tokenizer
from preliminary_experiments.data_generation.data_2_tree_search import GenerateTreeSearchData
from preliminary_experiments.data_generation.data_2_tree_search import GenerateTreeSearchDataRuntimeChecks
from preliminary_experiments.data_generation.data_utils import DataUtils
from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils


class DebugGenerateTreeSearchData(GenerateTreeSearchData):

    @classmethod
    def debug_print_traversal_history(cls, query, all_context, pred, answer, traversal_history, depth_history, instance):

        print("question:", query)
        print("pred:", pred, " answer:", answer)

        print("\n")

        for step_idx in range(len(instance["context_list"])):
            print(str(step_idx) + ": " + instance["context_list"][step_idx])

        print("\n")

        for step_idx in range(len(traversal_history)):
            print("\t" * depth_history[step_idx] + str(traversal_history[step_idx]) + ": " + all_context[
                traversal_history[step_idx]])

    @classmethod
    def debug_generate_grounded_statements(cls, with_initial_s=True):

        quants = list(range(cls.num_item_bound[1] + 1))

        for i in range(50):
            n_statements_to_gen = random.randint(3, 5)

            if with_initial_s:
                existing_grounded_chara_item = {(cls.full_names[0], cls.items[0]), (cls.full_names[1], cls.items[1])}
                existing_grounded_chara_quant_item = {(cls.full_names[0], quants[0], cls.items[0]),
                                                      (cls.full_names[1], quants[1], cls.items[1])}
                existing_ungrounded_chara_quant_item = {(cls.full_names[2], quants[2], cls.items[2]),
                                                        (cls.full_names[3], quants[3], cls.items[3])}
            else:
                existing_grounded_chara_item = set([])
                existing_grounded_chara_quant_item = set([])
                existing_ungrounded_chara_quant_item = set([])

            print("=" * 40)
            print("existing_g_c_i:")
            print(existing_grounded_chara_item)
            print("-" * 40)
            print("existing_g_c_q_i:")
            print(existing_grounded_chara_quant_item)
            print("-" * 40)
            print("existing_ung_c_q_i:")
            print(existing_ungrounded_chara_quant_item)
            print("-" * 40)

            grounded_statements, existing_grounded_chara_item, existing_grounded_chara_quant_item, \
                existing_ungrounded_chara_quant_item = GenerateTreeSearchData.generate_grounded_statements(
                    n_statements_to_gen,
                    existing_grounded_chara_item=existing_grounded_chara_item,
                    existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                    existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item)

            print("sampled:")
            print(grounded_statements)
            print("-" * 40)
            print("existing_g_c_i:")
            print(existing_grounded_chara_item)
            print("-" * 40)
            print("existing_g_c_q_i:")
            print(existing_grounded_chara_quant_item)
            print("-" * 40)
            print("existing_ung_c_q_i:")
            print(existing_ungrounded_chara_quant_item)
            input("-" * 40)

    @classmethod
    def check_generate_ungrounded_statements(cls):

        for i in range(50):
            n_statements_to_gen = 2
            existing_grounded_chara_item = set([])
            existing_grounded_chara_quant_item = set([])
            existing_ungrounded_chara_quant_item = {(cls.full_names[0], 3, cls.items[0])}

            grounded_statements, existing_grounded_chara_item, existing_grounded_chara_quant_item, \
                existing_ungrounded_chara_quant_item = GenerateTreeSearchData.generate_grounded_statements(
                n_statements_to_gen,
                existing_grounded_chara_item=existing_grounded_chara_item,
                existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item)

            print("=" * 40)
            print("existing_g_c_i:")
            print(existing_grounded_chara_item)
            print("-" * 40)
            print("existing_g_c_q_i:")
            print(existing_grounded_chara_quant_item)
            print("-" * 40)
            print("existing_ung_c_q_i:")
            print(existing_ungrounded_chara_quant_item)
            print("-" * 40)

            ungrounded_s_1var, existing_ungrounded_chara_quant_item = \
                GenerateTreeSearchData.generate_ungrounded_statements(
                    n_statements_to_gen, step_grounded_statements=grounded_statements,
                    existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                    existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item)

            print("sampled:")
            print(ungrounded_s_1var)
            print("-" * 40)
            print("existing_g_c_i:")
            print(existing_grounded_chara_item)
            print("-" * 40)
            print("existing_g_c_q_i:")
            print(existing_grounded_chara_quant_item)
            print("-" * 40)
            print("existing_ung_c_q_i:")
            print(existing_ungrounded_chara_quant_item)
            input("-" * 40)

    @classmethod
    def debug_statements_combination(cls):

        statements = [1, 2, 3, 4]

        for i in range(10):
            combinations = GenerateTreeSearchData.generate_statement_combination(statements)

            print(combinations)

    @classmethod
    def print_example_formal_reps(cls, instance):

        print("=" * 40)
        print(instance["question"])

        print("-" * 40)
        print("grounded statements")
        for s in instance["statements"]["grounded"][0]:
            print("\t", s)

        print("ungrounded statements")
        for s in instance["statements"]["ungrounded"][0]:
            print("\t", s)

        for d in range(instance["depth"]):
            print("-" * 40)

            print("grounded rules:")
            for s in instance["rules"]["grounded 1 var"][d] + instance["rules"]["grounded 2 var"][d]:
                print("\t", s)

            print("ungrounded rules:")
            for s in instance["rules"]["ungrounded 1 var"][d] + instance["rules"]["ungrounded 2 var"][d]:
                print("\t", s)

            print("grounded statements:")
            for s in instance["statements"]["grounded"][d + 1]:
                print("\t", s)

            print("ungrounded statements:")
            for s in instance["statements"]["ungrounded"][d + 1]:
                print("\t", s)

        input("-" * 40)

    @classmethod
    def debug_generate_one_structured_example(cls, depth=2, k=3):

        random.seed(depth)

        for i in range(100):
            instance = GenerateTreeSearchData.generate_one_structured_example(depth=depth, k=k, debug_flag=False)
            cls.print_example_formal_reps(instance)

    @classmethod
    def debug_generate_one_example(cls, depth=2, k=3):
        for i in range(100):
            instance = GenerateTreeSearchData.generate_one_example(
                depth=depth, k=k, tokenizer=T5Tokenizer.from_pretrained("t5-small"),
                initial_statements_grounded=set([]),
                initial_statements_ungrounded=set([]),
                existing_grounded_chara_item=set([]),
                existing_grounded_chara_quant_item=set([]),
                existing_ungrounded_chara_quant_item=set([]),
                names_to_subtract=set([]))
            print("=" * 40)
            print(json.dumps(instance, indent=2))
            input("-" * 40)

    @classmethod
    def debug_generated_tree_search_data(cls, depth=3, k=3):
        '''
        Things to check:
         - the person names of each step should have no overlap with previous names.
         -
        :return:
        '''

        random.seed(depth)

        for i in range(60000):
            instance = GenerateTreeSearchData.generate_one_structured_example(depth=depth, k=k, initial_statements=None,
                                                                              debug_flag=False)

            cls.debug_generated_tree_search_data_one_example(instance, depth)

    @classmethod
    def check_generated_tree_search_data_with_backward_chaining(cls, depth=2, k=3, debug_flag=False):
        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        random.seed(depth)

        n_true = 0
        n_false = 0

        n_statements_per_example = []
        n_rules_per_example = []
        for i in range(30000):

            instance = GenerateTreeSearchData.generate_one_example(
                depth=depth, k=k, tokenizer=tokenizer,
                initial_statements_grounded=set([]), initial_statements_ungrounded=set([]),
                debug_flag=False)

            if debug_flag:
                print("=" * 40)
                print(instance["question"], "\n")
                print("-" * 40)
                for s in instance["context_list"]:
                    print(s)
                print("-" * 40)

            pred, answer, all_statements, all_rules = \
                GenerateTreeSearchDataRuntimeChecks.runtime_check_generated_tree_search_data_with_backward_chaining_one_example(
                    instance, debug_flag=debug_flag)

            if debug_flag:
                print_instance = input("print instance?")
                if print_instance == "y" or print_instance == "Y":
                    print("-" * 40)
                    print(json.dumps(instance, indent=2))
                input("-" * 40)

            if answer:
                n_true += 1
            else:
                n_false += 1

            n_statements_per_example.append(len(all_statements))
            n_rules_per_example.append(len(all_rules))

            if i % 1000 == 0:
                print("processing instance ", i)

        print(n_true, n_false)
        print("avg num statements:", np.mean(n_statements_per_example))
        print("avg num rules:", np.mean(n_rules_per_example))

        # Seems to be good so far. No problem and the labels are evenly distributed.

    @classmethod
    def debug_natural_language_examples(cls, depth=2, k=3, debug_flag=True):

        random.seed(depth)

        for i in range(30000):
            instance = GenerateTreeSearchData.generate_one_structured_example(depth=depth,
                                                                              k=k,
                                                                              initial_statements=None)

            all_text, all_statements, all_rules, query = GenerateTreeSearchData.generate_natural_language_expressions_from_structured_example(instance)

            if debug_flag:
                print("=" * 40)
                for i, s in enumerate(all_statements):
                    print(i, s)

                print("\n")

                for i, r in enumerate(all_rules):
                    print(i, r)

                print("\n")
                print(query)

                input("-" * 40)

                print(instance["answer"])

                input("-" * 40)

    @classmethod
    def check_one_example(cls, du=2, split="train", depth=2):

        data_path = "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/tree_search_v0.5/tree_search_data_du" + str(
            du) + ".json"

        instances_all_splits = DataUtils.load_json(data_path)

        for instance in instances_all_splits[split]:
            if instance["depth"] == depth:
                print("=" * 40)
                print(instance["context_string"])
                print(instance["question_string"])
                print(instance["answer"])

                print("depth:", instance["depth"])

                input("-" * 40)

    @classmethod
    def check_data_statistics(cls, du=2):

        data_path = "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/tree_search_v0.2/" \
                    "tree_search_data_du" + str(du) + ".json"

        instances = DataUtils.load_json(data_path)

        for split in ["train", "dev", "test"]:

            n_pos = 0
            n_neg = 0

            for instance in instances[split]:
                if instance["answer"] != "I don't know":
                    n_pos += 1
                else:
                    n_neg += 1

            print("du:", du, " split:", split, " pos:", n_pos, " neg:", n_neg)


class DebugGenerateTreeSearchEVRData:

    @classmethod
    def debug_generate_evr_instances(cls,
                                     print_original_instance=False,
                                     print_evr_instance=False,
                                     check_depth=1):

        instances = DataUtils.load_json(
            "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/"
            "data_generated/tree_search_v0.5/tree_search_data_du5.json"
        )["train"]

        random.seed(0)
        random.shuffle(instances)

        # Count the number of generated example for each depth
        depth_to_num_count = {}
        for instance in instances:

            if print_original_instance or print_evr_instance:
                if instance["depth"] != check_depth:
                    continue

            (evr_instances, query_proved_flag, proof_chk_idx, statement_nl_chunks, rule_nl_chunks,
             traversal_history, depth_history) = cls.generate_evr_data_one_instance(instance)

            if instance["depth"] not in depth_to_num_count:
                depth_to_num_count[instance["depth"]] = []
            depth_to_num_count[instance["depth"]].append(len(evr_instances))

            if print_original_instance:
                print("=" * 40)
                all_chunks = statement_nl_chunks + rule_nl_chunks
                print("-" * 40)
                print(instance["question"])
                print("-" * 40)

                # Print the chunks of the instances.
                for chk_idx, chk in enumerate(all_chunks):
                    print(chk_idx, chk)
                print("-" * 40)

                # Print the traversal history of the origin problem
                for pr_idx in range(len(traversal_history)):
                    p_depth = depth_history[pr_idx]
                    chk_idx, s_idx = traversal_history[pr_idx]
                    print("\t" * p_depth, all_chunks[chk_idx][s_idx])
                print('-' * 40)

            if print_evr_instance:
                # Print the evr instance.
                for evr_instance in evr_instances:
                    print("-" * 40)
                    p_depth = evr_instance["search_depth"]
                    j_lines = json.dumps(evr_instance, indent=2).split("\n")
                    j_lines = ["\t" * p_depth + j_l for j_l in j_lines]
                    j_lines = "\n".join(j_lines)
                    print(j_lines)

            if print_original_instance or print_evr_instance:
                input("----")

        num_count = {k: sum(v) / len(v) for k, v in depth_to_num_count.items()}

        print(num_count)

        # By running this function, we also want to select a few examples that could be used to write test for.
        # Desired properties:
        # depth 0 proved
        # depth 0 not proved
        # depth 1 proved, ideally with two matched rules.
        # depth 1 not proved

    @classmethod
    def debug_check_instance_structure(cls):

        """
        This functions several aspects of the tree search instances, including whether the formal representations
        are aligned with the natural language representations.

        The format of each instance:
        {
            depth,
            provable: 0 or 1.
            statements: {
                "grounded": [
                    [name, num_items, items_tuple],
                ],
                "ungrounded: [],
                "distractors: [],
            }
            question: [name, num_items, items_tuple],
            answer: Yes/No
            context_list: [list of statements and rules in nl]
            context_string: a single string concatenated from the context list.
            question_string: from the structured questions
            statement_indices_shuffle_map:
            rule_indices_shuffle_map:
            id:
            context_len:
        }
        :return:
        """

        instances = DataUtils.load_json(
            "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/"
            "data_generated/tree_search_v0.5/tree_search_data_du5.json"
        )["train"]

        random.seed(0)

        random.shuffle(instances)

        for idx, instance in enumerate(instances):
            all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                             [s for step_s in instance["statements"]["distractors"] for s in step_s]
            # Only the initial grounded statement should be added

            all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                        [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                        [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                        [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                        [r for r in instance["rules"]["backtracking"] if r != None]

            # This way the order of the formal representations can be associated with the natural language representations
            all_statements = [all_statements[int(old_idx)]
                              for old_idx in instance["statement_indices_shuffle_map"].keys()]

            all_rules = [all_rules[int(old_idx)]
                         for old_idx in instance["rule_indices_shuffle_map"].keys()]

            all_statements_nl = instance["context_list"][:len(all_statements)]

            all_rules_nl = instance["context_list"][len(all_statements):]

            print("-" * 40)
            print(all_statements + all_rules)
            print("-" * 40)
            print(all_statements_nl + all_rules_nl)
            input("-" * 40)

    @classmethod
    def debug_check_splitting_chunks(cls):

        cases = [
            {
                "s": [1, 2, 3, 4, 5, 6],
                "r": [1, 2, 3, 4, 5, 6],
            },
            {
                "s": [1, 2, 3, 4, 5],
                "r": [1, 2, 3, 4, 5],
            },
            {
                "s": [1],
                "r": [1],
            },
            {
                "s": [1, 2, 3, 4, 5, 6],
                "r": [1, 2, 3, 4, 5],
            },
            {
                "s": [1, 2, 3, 4, 5],
                "r": [1, 2, 3, 4, 5, 6],
            },
        ]

        for case in cases:
            print("=" * 40)
            print(case)
            s_chunks, r_chunks = cls.split_to_chunks(case["s"], case["r"])
            print(s_chunks)
            print(r_chunks)
            input("-----")


if __name__ == "__main__":
    #DebugGenerateTreeSearchData.check_generated_tree_search_data_with_backward_chaining(debug_flag=True)
    DebugGenerateTreeSearchData.debug_generate_one_example(depth=1, k=3)
