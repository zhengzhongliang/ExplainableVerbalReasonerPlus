import pathlib
import hashlib
import os

from preliminary_experiments.data_generation.data_utils import DataUtils, DatasetUtils


class DataBase:

    items = [
        ("apple", "apples"), ("pear", "pears"), ("peach", "peaches"), ("banana", "bananas"),
        ("puppy", "puppies"), ("kitten", "kittens"), ("rabbit", "rabbits"), ("owl", "owls"),
        ("toy car", "toy cars"), ("toy bear", "toy bears"), ("pen", "pens"), ("ruler", "rulers"),
    ]

    full_names = DataUtils.combine_raw_first_last_names(num_name_to_use=100)

    operations = ["-2", "-1", "0", "1", "2"]

    num_item_bound = (1, 20)  # The person can have 1 to 20 items

    hash_module = hashlib.md5()

    # Get the data folder path
    file_path = pathlib.Path(__file__).resolve().parent
    project_script_folder_path = file_path.parent.parent
    project_folder_path = project_script_folder_path.parent

    project_data_folder_path = os.path.join(project_folder_path, "data")
    if not os.path.exists(project_data_folder_path):
        os.mkdir(project_data_folder_path)

