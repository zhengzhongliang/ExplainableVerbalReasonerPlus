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


class CheckIfHandler:

    cases = [
        {
            "lines": [
                "#1 = 1",
                "#2 = 2",
                "if #1 > 0",
                    "#1 = arith_sum( #1, 1)",
                "else",
                    "#2 = arith_sum( #2, 1)",
                "end_if",
            ],
            "lvd": {},
            "ebd": {},
            "emd": {},
        },

        {
            "lines": [
                "#1 = 0",
                "#2 = 2",
                "if #1 > 0",
                    "#1 = arith_sum( #1, 1)",
                "else",
                    "#2 = arith_sum( #2, 1)",
                "end_if",
            ],
            "lvd": {},
            "ebd": {},
            "emd": {},
        },

        {
            "lines": [
                "#1 = 1",
                "#2 = 2",
                "#3 = 3",
                "#4 = 4",
                "if #1 > 0",
                    "#1 = arith_sum( #1, 1)",
                    "if #2 > 0",
                        "#4 = arith_sum( #4, 1)",
                    "else",
                        "#4 = arith_sum( #4, -1)",
                    "end_if",
                "else",
                    "#2 = arith_sum( #2, 1)",
                    "if #2 > 0",
                        "#4 = arith_sum( #4, 1)",
                    "else",
                        "#4 = arith_sum( #4, -1)",
                    "end_if",
                "end_if",
                "#3 = arith_sum( #3, 1)",
            ],

            "lvd": {},
            "ebd": {},
            "emd": {},
        },

        {
            "lines": [
                "#1 = 0",
                "#2 = 2",
                "#3 = 3",
                "#4 = 4",
                "if #1 > 0",
                    "#1 = arith_sum( #1, 1)",
                    "if #2 > 0",
                        "#4 = arith_sum( #4, 1)",
                    "else",
                        "#4 = arith_sum( #4, -1)",
                    "end_if",
                "else",
                    "#2 = arith_sum( #2, 1)",
                    "if #2 > 5",
                        "#4 = arith_sum( #4, 1)",
                    "else",
                        "#4 = arith_sum( #4, -1)",
                    "end_if",
                "end_if",
                "#3 = arith_sum( #3, 1)",
            ],

            "lvd": {},
            "ebd": {},
            "emd": {},
        },
    ]

    prog_int = ProgramInterpreter(neural_module=NeuralModule())

    @classmethod
    def check_get_if_span(cls):
        program = cls.cases[0]["lines"]

        if_start, if_else, if_end, if_programs = cls.prog_int.get_if_span(2, program)

        print(if_start, if_else, if_end)
        print(if_programs)

    @classmethod
    def check_get_if_span_nested(cls):
        program = cls.cases[0]["lines"]

        if_block_start, if_block_else, if_block_end, if_programs = cls.prog_int.get_if_span(4, program)

        print(if_block_start, if_block_else, if_block_end)
        print(if_programs)

        if_condition_judgement_line = program[if_block_start]

        if_body = program[if_block_start + 1: if_block_else]

        else_body = program[if_block_else + 1: if_block_end]

        print("if condition:", if_condition_judgement_line)
        print("if body:", if_body)
        print("else body:", else_body)

    @classmethod
    def check_if_handler(cls):

        for case in cls.cases:
            print("=" * 40)
            print("lines:", case["lines"])
            print("lvd:", case["lvd"])
            print("ebd:", case["ebd"])
            print("emd:", case["emd"])
            print("-" * 40)

            case["lvd"] = cls.prog_int.program_handler(case["lines"], case["lvd"], case["ebd"], case["emd"])

            print("lvd:", case["lvd"])
            print("program counter:")

            print("=" * 40)

if __name__ == "__main__":

    CheckIfHandler.check_if_handler()
