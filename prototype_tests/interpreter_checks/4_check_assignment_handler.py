from preliminary_experiments.experiments_evr.evr_class.InterpreterClass import ProgramInterpreter


class NeuralModule:

    def __init__(self):
        pass

    def inference(self, textual_input):

        '''
        Need to to a few pure rule based output to make sure the whole workflow is fine
        :param textual_input:
        :return:
        '''

        # TODO: add rules here later
        return textual_input


class CheckAssignmentHandler:

    @classmethod
    def check_assign_handler(cls):

        prog_int = ProgramInterpreter(neural_module=NeuralModule())

        cases = [
            {
                "lines": [" #1  = 2"],
                "lvd": {},
                "ebd": {},
                "emd": {},
            },

            {
                "lines": [" #1 = 'abc' "],
                "lvd": {},
                "ebd": {},
                "emd": {},
            },

            {
                "lines": [" #1 = None  "],
                "lvd": {},
                "ebd": {},
                "emd": {},
            },

            {
                "lines": [" #1 = True"],
                "lvd": {},
                "ebd": {},
                "emd": {},
            },

            {
                "lines": [" #2 = #1"],
                "lvd": {"#1": "abc"},
                "ebd": {},
                "emd": {},
            },

            {
                "lines": [" #2 = episodic_buffer_1"],
                "lvd": {"#1": "abc"},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {},
            },

            {
                "lines": [" #3 = qa(#1, episodic_buffer_1)"],
                "lvd": {"#1": "abc"},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {},
            },

            {
                "lines": [" #3 = rewrite(#1, episodic_buffer_1)"],
                "lvd": {"#1": "abc"},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {},
            },

            {
                "lines": [" #3 = subq( )"],
                "lvd": {"#1": "abc"},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {},
            },

            {
                "lines": [" #3 = get_chunk( #1 ) "],
                "lvd": {"#1": 0},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {0: 1234},
            },

            {
                "lines": [" #3 = get_chunk( #1 ) "],
                "lvd": {"#1": "0"},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {"0": 1234},
            },

            {
                "lines": [" #3 = arith_sum( #1 , #2) "],
                "lvd": {"#1": -1, "#2": 5},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {"0": 1234},
            },

            {
                "lines": [" #2 = arith_sum( #1 , #2) "],
                "lvd": {"#1": -1, "#2": 5},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {"0": 1234},
            },

            {
                "lines": [" #1 = arith_sum( #1 , 1) "],
                "lvd": {"#1": -1, "#2": 5},
                "ebd": {"episodic_buffer_1": "xyz"},
                "emd": {"0": 1234},
            },

        ]

        for case in cases:
            print("=" * 40)
            print("lines:", case["lines"])
            print("lvd:", case["lvd"])
            print("ebd:", case["ebd"])
            print("emd:", case["emd"])
            print("-" * 40)

            program_counter, case["lvd"] = prog_int.assign_handler(case["lines"], 0, case["lvd"], case["ebd"], case["emd"])

            print("lvd:", case["lvd"])
            print("program counter:", program_counter)

            print("=" * 40)


if __name__ == "__main__":
    CheckAssignmentHandler.check_assign_handler()