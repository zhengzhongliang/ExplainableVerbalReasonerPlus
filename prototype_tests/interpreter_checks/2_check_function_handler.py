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


class CheckFunctionHandler:

    @classmethod
    def check_qa(cls):

        prog_int = ProgramInterpreter(neural_module=NeuralModule())

        snippets = [
            {
                "args": ["  #1  ", "   #2    "],
                "l": {"#1": 1, "#2": "1234"},
                "e": {}
            },
            {
                "args": ["  #1  ", "   episodic_buffer_1    "],
                "l": {"#1": 1, "#2": "1234"},
                "e": {"episodic_buffer_1": "1234"}
            }
        ]

        print("=" * 40)
        print("check qa handler")
        print("=" * 40)

        for s in snippets:
            print(prog_int.qa_handler(s["args"], s["l"], s["e"], {}))
            print("-" * 40)

    @classmethod
    def check_rewrite(cls):

        prog_int = ProgramInterpreter(neural_module=NeuralModule())

        snippets = [
            {
                "args": ["  #1  ", "   #2    "],
                "l": {"#1": 1, "#2": "1234"},
                "e": {}
            },
            {
                "args": ["  #1  ", "   episodic_buffer_1    "],
                "l": {"#1": 1, "#2": "1234"},
                "e": {"episodic_buffer_1": "1234"}
            }
        ]

        print("=" * 40)
        print("check rewrite handler")
        print("=" * 40)

        for s in snippets:
            print(prog_int.rewrite_handler(s["args"], s["l"], s["e"], {}))
            print("-" * 40)

    @classmethod
    def check_subq(cls):

        prog_int = ProgramInterpreter(neural_module=NeuralModule())

        snippets = [
            {
                "args": ["  #1  ", "   #2    "],
                "l": {"#1": 1, "#2": "1234"},
                "e": {"episodic_buffer_1": "1234", "episodic_buffer_2": 5678}
            },
            {
                "args": ["  #1  ", "   episodic_buffer_1    "],
                "l": {"#1": 1, "#2": "1234"},
                "e": {"episodic_buffer_1": "1234"}
            }
        ]

        print("=" * 40)
        print("check subq handler")
        print("=" * 40)

        for s in snippets:
            print(prog_int.subq_handler(s["args"], s["l"], s["e"], {}))
            print("-" * 40)


if __name__ == "__main__":

    CheckFunctionHandler.check_qa()
    CheckFunctionHandler.check_rewrite()
    CheckFunctionHandler.check_subq()



