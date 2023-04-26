import json
import pickle


class DataReadWriteUtils:

    @classmethod
    def load_jsonl(cls, file_dir):
        with open(file_dir, "r") as handle:
            data = [json.loads(line) for line in handle]

        return data

    @classmethod
    def load_json(cls, file_path):

        with open(file_path, "r") as handle:
            json_item = json.load(handle)

        return json_item

    @classmethod
    def write_json(cls, json_item, file_path):

        with open(file_path, "w") as handle:
            json.dump(json_item, handle)

    @classmethod
    def load_pickle(cls, file_path):
        with open(file_path, "rb") as handle:
            pickle_item = pickle.load(handle)

        return pickle_item

    @classmethod
    def write_pickle(cls, pickle_item, file_path):
        with open(file_path, "wb") as handle:
            pickle.dump(pickle_item, handle)
