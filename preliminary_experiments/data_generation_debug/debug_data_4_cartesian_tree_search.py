import random
import numpy as np
import json

from transformers import T5Tokenizer

from preliminary_experiments.data_generation.data_4_cartesian_tree_search import GenerateCartesianTreeSearchData
from preliminary_experiments.data_generation.data_utils import DataUtils


class DebugGenerateCartesianTreeSearchData(GenerateCartesianTreeSearchData):

    @classmethod
    def print_example(cls, instance):
        """Print one generated example.

        The structure of the generated example:
        {
            cartesian_instance: {
                "id": instance_id,"depth": depth,
                "context_string": context,
                "question_string": question,
                "answer": target_statement_nl,
                "target_list": target_statement_list,
                "target_nl_list": target_statement_nl_list,
                "ungrounded_list": ungrounded_statement_list,
                "ungrounded_nl_list": ungrounded_statement_nl_list,
                "target_len": len_tokenized_target

            },
            tree_search_instance: {
                "depth": depth,
                "provable": instance_label,
                "statements": generated_statements,
                "rules": sampled_rules,
                "question": question_triple,
                "answer": answer,
                "context_list": all statements and rules, without question
                "context_string": the single string of all statements in the context list
                "question_string": natural language question
                "statement_indices_shuffle_map":
                "rule_indices_shuffle_map":
                "target_text_w_inter":
            },
            "context_string":
            "question_string": question string of the tree search task
            "target_text":
            "target_text_w_inter":
        }
        :param depth:
        :param k:
        :return:
        """

        print("=" * 40)

        print("context_list:")
        context_list_ = instance["context_string"].split(". ")
        for c_ in context_list_:
            print(f"\t{c_}")

        print("-" * 40)
        print("question:", instance["question_string"])

        print("-" * 40)
        print("cartesian target list all:")
        target_list_all = instance["cartesian_instance"]["target_list"]
        for t_ in target_list_all:
            print(f"\t{t_}")

        print("-" * 40)
        print("cartesian target list ungrounded:")
        target_list_ungrounded = instance["cartesian_instance"]["ungrounded_list"]
        for t_ in target_list_ungrounded:
            print(f"\t{t_}")

        print("-" * 40)
        print("target with proof:")
        target_w_proof = instance["target_text_w_inter"].split(". ")
        for t_ in target_w_proof:
            print(f"\t{t_}")

        input("----")

    @classmethod
    def check_generate_one_example(cls, depth=2, k=3):
        random.seed(depth)

        for i in range(100):
            instance = GenerateCartesianTreeSearchData.generate_one_example(depth=depth, k=k, debug_flag=False)

            #cls.print_example(instance)
            print(instance)

    @classmethod
    def convert_list_to_tuple_recursive(cls, l):
        return tuple(cls.convert_list_to_tuple_recursive(x) for x in l) if type(l) is list else l

    @classmethod
    def check_all_grounded_ungrounded_no_overlap(cls, du):

        file_path = ("/Users/curry/zhengzhong/research/2022_NLTuringMachine/data/"
                     f"cartesian_tree_search_v1.0/cartesian_tree_search_data_du{du}.json")

        with open(file_path, "r") as handle:
            instances = json.load(handle)

        for split in ["train", "dev", "test"]:
            for instance in instances[split]:
                grounded = set(cls.convert_list_to_tuple_recursive(instance["cartesian_instance"]["target_list"]))
                for s_grounded in instance["tree_search_instance"]["statements"]["grounded"]:
                    grounded.update(set(cls.convert_list_to_tuple_recursive(s_grounded)))

                ungrounded = set(cls.convert_list_to_tuple_recursive(instance["cartesian_instance"]["ungrounded_list"]))
                for s_ungrounded in instance["tree_search_instance"]["statements"]["ungrounded"]:
                    ungrounded.update(set(cls.convert_list_to_tuple_recursive(s_ungrounded)))

                assert len(grounded.intersection(ungrounded)) == 0


if __name__ == "__main__":
    DebugGenerateCartesianTreeSearchData.check_all_grounded_ungrounded_no_overlap(du=1)
    DebugGenerateCartesianTreeSearchData.check_all_grounded_ungrounded_no_overlap(du=4)