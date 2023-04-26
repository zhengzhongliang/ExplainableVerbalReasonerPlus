from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent
import json

class NeuralModule:

    def __init__(self):
        pass

    def inference(self, textual_input):

        '''
        Need to to a few pure rule based output to make sure the whole workflow is fine
        :param textual_input:
        :return:
        '''

        return textual_input


class CheckRewriteHandler:

    evr_agent = EVRAgent(neural_module=NeuralModule())

    cases = [
        {
            "args": ["#1", "#2"],
            "l": {
                "#1": "can this chunk to be used to prove it?",
                "#2": "False"
            },
            "e": {}
        },
        {
            "args": ["#1", "#2", "#3"],
            "l": {
                "#1": "how many items did this person have?",
                "#2": "John Smith",
                "#3": "2 toys"
            },
            "e": {}
        },
        {
            "args": ["#1"],
            "l": {
                "#1": ["John Smith had 2 toys. ", "John Due had 3 apples."]
            },
            "e": {}
        },
    ]

    @classmethod
    def check_rewrite(cls):

        for case in cls.cases:
            print("=" * 40)
            print(json.dumps(case, indent=2))
            print(cls.evr_agent.rewrite_handler(case["args"], case["l"], case["e"], {}))


if __name__ == "__main__":
    CheckRewriteHandler.check_rewrite()
