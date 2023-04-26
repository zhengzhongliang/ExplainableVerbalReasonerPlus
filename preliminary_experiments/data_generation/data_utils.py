import json
import numpy as np
import pandas as pd

from pathlib import Path
from transformers import T5Tokenizer


class DataUtils:
    '''
    This class contains the code that could be used later to generate the datasets
    '''

    raw_names_path = Path(__file__).parent / 'data/raw_names/raw_names.xlsx'

    @classmethod
    def load_json(cls, file_dir):

        with open(file_dir, "r") as handle:
            result = json.load(handle)

        return result

    @classmethod
    def load_jsonl(cls, file_dir):

        with open(file_dir, "r") as handle:
            data = [json.loads(line) for line in handle]

        return data

    @classmethod
    def read_raw_names(cls, file_path=None):
        '''
        Get the name from: https://namecensus.com/
        This method is used in the "Chain of Thought Prompting Elicits Reasoning in Large Language Models" paper.

        We temporarily decide to use 1000 first names and 1000 last names.
        '''

        if not file_path:
            file_path = cls.raw_names_path

        df_first_name = pd.read_excel(file_path, sheet_name="first_name")
        df_last_name = pd.read_excel(file_path, sheet_name="last_name")

        first_names = list(df_first_name.iloc[:, 1])  # 1000 first names, from the most frequent to the least
        last_names = list(df_last_name.iloc[:, 1])  # 1000 last names, from the most frequent to the least

        first_names = [name[0] + name[1:].lower() for name in first_names]
        last_names = [name[0] + name[1:].lower() for name in last_names]

        return first_names, last_names

    @classmethod
    def combine_raw_first_last_names(cls, num_name_to_use=100):
        '''
        This function combines the top first names and last names to make more names.
        :param num_name_to_use:
        :return:
        '''
        assert num_name_to_use <= 1000, "don't have so many candidate names!"

        first_names, last_names = cls.read_raw_names()

        full_names = []
        for fn in first_names[:num_name_to_use]:
            for ln in last_names[:num_name_to_use]:
                full_names.append(fn + " " + ln)

        return full_names

    @classmethod
    def get_all_names(cls):
        full_names = DataUtils.combine_raw_first_last_names()

        return full_names


class TestDataUtils:

    @classmethod
    def test_read_raw_names(cls):
        first_names, last_namess = DataUtils.read_raw_names()

        print(first_names)

    @classmethod
    def test_combine_raw_first_last_names(cls):
        full_names = DataUtils.combine_raw_first_last_names()
        full_names_dict = {fn: 0 for fn in full_names}

        print(full_names)

        print("=" * 40)

        # Check basic statistics
        print("num of full names generated:", len(full_names))  # 10,000
        print("num of unique names:", len(full_names_dict))   # 10,000

        # Check whether there are special characters in the full name list
        special_chars = [".", ";", ",", "-", "/", "?"]

        special_names = {}
        for special_char in special_chars:
            for fn in full_names:
                if special_char in fn:
                    special_names[fn] = 1
        if len(special_names) == 0:
            print("no special characters in name")
        else:
            print("names with special characters:")
            print(special_names)


class DatasetUtils:

    @classmethod
    def get_dataset_depth_statistics(cls, instances):
        splits = ["train", "dev", "test"]

        statistics = {split: {} for split in splits}
        for split in splits:
            for instance in instances[split]:
                if instance["depth"] not in statistics[split]:
                    statistics[split][instance["depth"]] = 1
                else:
                    statistics[split][instance["depth"]] += 1

            statistics[split]["all"] = sum(statistics[split].values())

        return statistics

    @classmethod
    def get_dataset_answer_statistics(cls, instances):
        splits = ["train", "dev", "test"]

        statistics = {split: {} for split in splits}
        for split in splits:
            for instance in instances[split]:
                if instance["answer"] not in statistics[split]:
                    statistics[split][instance["answer"]] = 1
                else:
                    statistics[split][instance["answer"]] += 1

            statistics[split]["all"] = sum(statistics[split].values())

        return statistics

    @classmethod
    def get_dataset_input_length_statistics(cls, instances, tokenizer=None):

        # Use the t5 tokenizer by default.
        if tokenizer == None:
            tokenizer = T5Tokenizer.from_pretrained("t5-large", truncation=False)

        splits = ["train", "dev", "test"]

        statistics = {
            split: {"<512": 0, "512~1024": 0, ">1024": 0}
            for split in splits
        }
        for split in splits:
            for instance in instances[split]:
                context_question_string = instance["context_string"] + instance["question_string"]
                encoded_ids = tokenizer(context_question_string)["input_ids"]

                input_len = len(encoded_ids)

                if input_len < 512:
                    statistics[split]["<512"] += 1
                elif input_len < 1024:
                    statistics[split]["512~1024"] += 1
                else:
                    statistics[split][">1024"] += 1

            statistics[split]["all"] = sum(statistics[split].values())

        return statistics

    @classmethod
    def get_dataset_target_length_statistics(cls, instances, tokenizer=None):

        # Use the t5 tokenizer by default.
        if tokenizer == None:
            tokenizer = T5Tokenizer.from_pretrained("t5-large", truncation=False)

        splits = ["train", "dev", "test"]

        statistics = {
            split: {"<512": 0, "512~1024": 0, ">1024": 0}
            for split in splits
        }
        for split in splits:
            for instance in instances[split]:
                target_text = instance["answer"]
                encoded_ids = tokenizer(target_text)["input_ids"]

                input_len = len(encoded_ids)

                if input_len < 512:
                    statistics[split]["<512"] += 1
                elif input_len < 1024:
                    statistics[split]["512~1024"] += 1
                else:
                    statistics[split][">1024"] += 1

            statistics[split]["all"] = sum(statistics[split].values())

        return statistics

    @classmethod
    def get_dataset_statistics(cls, instances):
        statistics = {
            "depth": cls.get_dataset_depth_statistics(instances),
            "answer": cls.get_dataset_answer_statistics(instances),
            "input_len": cls.get_dataset_input_length_statistics(instances)
        }

        return statistics

    @classmethod
    def get_dataset_statistics_cartesian(cls, instances):
        statistics = {
            "depth": cls.get_dataset_depth_statistics(instances),
            "input_len": cls.get_dataset_input_length_statistics(instances),
            "target_len": cls.get_dataset_target_length_statistics(instances)
        }

        return statistics


if __name__ == "__main__":
    TestDataUtils.test_read_raw_names()
    #TestDataUtils.test_combine_raw_first_last_names()






