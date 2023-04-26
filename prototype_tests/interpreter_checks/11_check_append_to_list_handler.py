from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent
import json


class CheckAppendToListHandler:

    evr_agent = EVRAgent(neural_module=None)

    cases = [
        {
            "args": ["#1", "#2"],
            "l": {
                "#1": ["A has x toys."],
                "#2": "B has y toys. "
            },
            "e": {}
        },
    ]

    @classmethod
    def check_append_to_list(cls):

        for case in cls.cases:
            print("=" * 40)
            print(json.dumps(case, indent=2))
            print(cls.evr_agent.append_to_list_handler(case["args"], case["l"], case["e"], {}))


if __name__ == "__main__":
    CheckAppendToListHandler.check_append_to_list()
