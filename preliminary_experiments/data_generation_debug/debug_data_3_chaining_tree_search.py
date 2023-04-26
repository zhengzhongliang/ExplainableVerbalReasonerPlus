import random
import numpy as np
import json

from transformers import T5Tokenizer

from preliminary_experiments.data_generation.data_3_chaining_tree_search import GenerateChainingTreeSearchData
from preliminary_experiments.data_generation.data_utils import DataUtils


class DebugGenerateChainingTreeSearchData(GenerateChainingTreeSearchData):

    @classmethod
    def print_example(cls, instance):

        # TODO: the function is not longer valid. Due to the change of the instances, the function is no longer usable.

        print('=' * 40)

        for i, chain in enumerate(instance["chaining_instance"]):
            print("-" * 40)
            print("chaining instance ", i)
            for c in chain["context_list"]:
                print("\t", c)

        print("-" * 40)
        print("tree search instance")
        for s in instance["tree_search_instance"]["context_list"]:
            print("\t", s)

        print("-" * 40)
        print("tree search answer:", instance["answer"])

        print("-" * 40)
        print("all statements")
        for s in instance["context_list"]:
            print("\t", s)

        print('\n')
        print("question")
        print(instance["question_string"])

        print("\n")
        print("answer:", instance["answer"])

    @classmethod
    def check_generate_one_example(cls, depth, k=2):
        """Check the generated example.

        The structure of the generated example:
        {
            chaining_instance: {
                "id": instance_id,
                "chains": [
                    {
                        "formal_reps": formal_reps_one_chain,
                        "quantity_ops": quantity_ops_one_chain,
                        "context_list": instance_contexts_one_chain,
                        "answer": num_buffer_one_chain
                    }
                ],
                "selected_chain_idx": selected_question_chain_idx,
                "context_string": concatenated_context,
                "question_string": question_string,
                "answer": selected_question_answer,
                "depth": depth
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

        random.seed(depth)

        for i in range(100):
            instance = GenerateChainingTreeSearchData.generate_one_example(depth=depth, k=k, debug_flag=False)
            #cls.print_example(instance)

            print("=" * 40)
            context_list_ = instance["context_string"].split(". ")
            context_list_ = [f"{c_}." for c_ in context_list_]

            target_list_ = instance["target_text_w_inter"].split(". ")
            target_list_ = [f"{t_}" for t_ in target_list_]

            print("context_list")
            for c_ in context_list_:
                print("\t", c_)

            print("-" * 40)
            print("question:", instance["question_string"])

            print("-" * 40)
            print("target with proof")
            for t_ in target_list_:
                print("\t", t_)

            input("-" * 40)

    @classmethod
    def load_data_and_check(cls):

        du = 5
        data_path = ("/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/"
                     f"chaining_tree_search_v0.1/tree_search_data_du{du}.json")

        data = DataUtils.load_json(data_path)

        tokenizer = T5Tokenizer.from_pretrained("t5-small")

        for instance in data["train"] + data["dev"] + data["test"]:
            encoded_ids = tokenizer(instance["context_string"], truncation=True, max_length=2048)["input_ids"]

            if len(encoded_ids) > 1024:
                print("!")

    @classmethod
    def check_one_example(cls, du=2, split="train", depth=2):

        data_path = ("/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/"
                     f"chaining_tree_search_v0.1/chaining_tree_search_data_du{du}.json")

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
    def check_instance_length(cls, du=5):
        data_path = ("/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/"
                     f"chaining_tree_search_v0.3/chaining_tree_search_data_du{du}.json")

        instances_all_splits = DataUtils.load_json(data_path)

        tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")

        lens = {}
        for split in ["train", "dev", "test"]:
            for instance in instances_all_splits[split]:
                input_str = instance["context_string"] + " " + instance["question_string"]

                input_len = len(tokenizer(input_str)["input_ids"])

                if instance["depth"] not in lens:
                    lens[instance["depth"]] = [input_len]
                else:
                    lens[instance["depth"]].append(input_len)

        for d in range(6):
            print("-" * 40)
            print("depth:", d)
            print("max:", max(lens[d]), " min:", min(lens[d]))
            print("mean:", np.mean(lens[d]), " std:", np.std(lens[d]))

    @classmethod
    def check_generate_one_example_raw(cls, depth=2, k=3):

        for i in range(100):
            instance = GenerateChainingTreeSearchData.generate_one_example(
                depth=depth,
            )

            print("=" * 40)
            print(json.dumps(instance))
            input("-" * 40)

    @classmethod
    def convert_list_to_tuple_recursive(cls, l):
        return tuple(cls.convert_list_to_tuple_recursive(x) for x in l) if type(l) is list else l

    @classmethod
    def check_all_grounded_ungrounded_no_overlap(cls, du):

        file_path = ("/Users/curry/zhengzhong/research/2022_NLTuringMachine/data/"
                     f"chaining_tree_search_v1.0/chaining_tree_search_data_du{du}.json")

        with open(file_path, "r") as handle:
            instances = json.load(handle)

        for split in ["train", "dev", "test"]:
            for instance in instances[split]:
                grounded = set(cls.convert_list_to_tuple_recursive(instance["initial_s_grounded"]))
                for s_grounded in instance["tree_search_instance"]["statements"]["grounded"]:
                    grounded.update(set(cls.convert_list_to_tuple_recursive(s_grounded)))

                ungrounded = set(cls.convert_list_to_tuple_recursive(instance["initial_s_ungrounded"]))
                for s_ungrounded in instance["tree_search_instance"]["statements"]["ungrounded"]:
                    ungrounded.update(set(cls.convert_list_to_tuple_recursive(s_ungrounded)))

                assert len(grounded.intersection(ungrounded)) == 0


if __name__ == "__main__":
    DebugGenerateChainingTreeSearchData.check_generate_one_example(depth=1, k=3)
    #DebugGenerateChainingTreeSearchData.check_generate_one_example_raw(depth=0)
    #DebugGenerateChainingTreeSearchData.check_all_grounded_ungrounded_no_overlap(du=2)
    #DebugGenerateChainingTreeSearchData.check_all_grounded_ungrounded_no_overlap(du=4)
