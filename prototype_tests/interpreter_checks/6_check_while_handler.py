from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent


class CheckWhileHandler:

    interpreter = EVRAgent(neural_module=None)

    programs = {
        "while": [
            "a = 1",
            "b = 2",
            "while a < 3",
            "a = a + 1",
            "end_while",
            "c = c + 1"
        ],
        "while_nested": [
            "a = 1",
            "b = 2",
            "while a < 3",
                "a = a + 1",
                "while b < 5",
                    "b = b + 1",
                "end_while",
            "end_while",
            "c = c + 1"
        ],
    }


    @classmethod
    def check_get_while_span(cls):
        program = cls.programs["while"]

        while_start, while_end, while_programs = cls.interpreter.get_while_span(2, program)

        print(while_start, while_end)
        print(while_programs)

        while_judge_condition = program[while_start]

        while_loop_body = program[while_start + 1: while_end]

        print("condition:", while_judge_condition)
        print("loop body:", while_loop_body)

    @classmethod
    def check_get_while_span_nested(cls):
        program = cls.programs["while_nested"]

        while_start, while_end, while_programs = cls.interpreter.get_while_span(2, program)

        print(while_start, while_end)
        print(while_programs)

        while_judge_condition = program[while_start]

        while_loop_body = program[while_start + 1: while_end]

        print("condition:", while_judge_condition)
        print("loop body:", while_loop_body)


if __name__ == "__main__":
    #CheckWhileHandler.check_get_while_span()
    CheckWhileHandler.check_get_while_span_nested()
    #CheckWhileHandler.check_get_if_span_nested()
