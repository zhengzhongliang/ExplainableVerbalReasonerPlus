from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent
import re


class NeuralModule:

    def __init__(self):
        pass

    def inference(self, textual_input):

        '''
        Need to do a few pure rule based output to make sure the whole workflow is fine
        :param textual_input:
        :return:
        '''

        return_var = ""

        if textual_input.startswith("qa: "):

            if "who are the persons" in textual_input:
                return_var = "the persons are ['John Smith', 'John Doe']. "

            if "what are the items" in textual_input:
                return_var = "the items are ['2 toys', '3 apples']. "

        if textual_input.startswith("clear_mem"):

            textual_input = textual_input.replace("clear_mem: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]
            textual_input_cleaned = " ".join(textual_list).lower()

            if len(textual_list) == 6 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the persons are ['" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned and \
                "#0 stores the list of persons" in textual_input_cleaned:

                return_var = \
                    "episodic_buffer_0: 'this is a cartesian task' " + \
                    "episodic_buffer_1: 'chunk 0 can be used to infer the number of items each person had.' " + \
                    "episodic_buffer_2: 'list the items each person had' " \
                    "episodic_buffer_3: 'the items are ['2 toys', '3 apples']' " \
                    "episodic_buffer_4: '#0 stores the list of persons' "

            if len(textual_list) == 6 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned and \
                "#0 stores the list of persons" in textual_input_cleaned and \
                "#1 stores the list of items" in textual_input_cleaned:

                return_var = \
                    "episodic_buffer_0: 'this is a cartesian task' " + \
                    "episodic_buffer_1: 'chunk 0 can be used to infer the number of items each person had.' " + \
                    "episodic_buffer_2: 'list the items each person had' " \
                    "episodic_buffer_3: '#0 stores the list of persons' " + \
                    "episodic_buffer_4: '#1 stores the list of items' "

        if textual_input.startswith("rewrite: "):
            if "how many items this person had?" in textual_input:
                statement = textual_input.replace("how many items this person had?", "")
                statements = [s for s in statement.split(".") if not s.isspace() and s != ""]
                return_var = " had ".join(statements) + ". "

            if "John Smith had" in textual_input:
                return_var = textual_input.replace("rewrite: ", "")

        if textual_input.startswith("generate_program"):

            textual_input = textual_input.replace("generate_program: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]
            textual_input_cleaned = " ".join(textual_list).lower()

            program_text = ""
            if len(textual_list) == 3 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned:

                program_text = \
                    "#0 = 'who are the persons';" + \
                    "new_mem(episodic_buffer_1, #0); "

            if len(textual_list) == 2 and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "who are the persons" in textual_input_cleaned:

                program_text = \
                    "#0 = get_chunk('chunk_0'); " + \
                    "#1 = qa(#0, episodic_buffer_1); " + \
                    "add_to_episodic(#1); "

            if len(textual_list) == 3 and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "who are the persons" in textual_input_cleaned and \
                "the persons are ['" in textual_input_cleaned:

                program_text = \
                    "return(episodic_buffer_2); "

            if len(textual_list) == 4 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the persons are ['" in textual_input_cleaned:

                program_text = \
                    "#0 = 'what are the items'; " + \
                    "new_mem(episodic_buffer_1, #0); "

            if len(textual_list) == 2 and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "what are the items" in textual_input_cleaned:

                program_text = \
                    "#0 = get_chunk('chunk_0'); " + \
                    "#1 = qa(#0, episodic_buffer_1); " + \
                    "add_to_episodic(#1); "

            if len(textual_list) == 3 and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "what are the items" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned:
                program_text = \
                    "return(episodic_buffer_2); "

            if len(textual_list) == 5 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the persons are ['" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned:

                program_text = \
                    "#0 = ['John Smith', 'John Doe']; " + \
                    "add_to_episodic('#0 stores the list of persons'); "

            if len(textual_list) == 6 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the persons are ['" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned and \
                "#0 stores the list of persons" in textual_input_cleaned:

                program_text = \
                    "clear_mem(); "

            if len(textual_list) == 5 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned and \
                "#0 stores the list of persons" in textual_input_cleaned:

                program_text = \
                    "#1 = ['2 toys', '3 apples']; " + \
                    "add_to_episodic('#1 stores the list of items'); "

            if len(textual_list) == 6 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "the items are ['" in textual_input_cleaned and \
                "#0 stores the list of persons" in textual_input_cleaned and \
                "#1 stores the list of items" in textual_input_cleaned:

                program_text = "clear_mem(); "

            if len(textual_list) == 5 and \
                "this is a cartesian task" in textual_input_cleaned and \
                "chunk 0 can be used to infer the number of items each person had" in textual_input_cleaned and \
                "list the items each person had" in textual_input_cleaned and \
                "#0 stores the list of persons" in textual_input_cleaned and \
                "#1 stores the list of items" in textual_input_cleaned:

                program_text = \
                    "#2 = []; " + \
                    "for #3 in #0; " + \
                        "for #4 in #1; " + \
                            "#5 = 'how many items this person had?'; " + \
                            "#6 = rewrite(#5, #3, #4); " + \
                            "#2 = append_to_list(#2, #6); " + \
                        "end_for; " + \
                    "end_for; " + \
                    "#7 = rewrite(#2); " + \
                    "add_to_episodic(#7);"

            return_var = program_text

        return return_var


class CheckCartesian:

    prog_int = EVRAgent(neural_module=NeuralModule())

    @classmethod
    def check_cartesian_integrated(cls):

        external_chunks = {
            "chunk_0": {
                "statement_0": "each of John Smith and John Doe had 2 toys and 3 apples. "
            }
        }

        episodic_buffer_dict = {
            "episodic_buffer_0": "this is a cartesian task. ",
            "episodic_buffer_1": "chunk 0 can be used to infer the number of items each person had. ",
            "episodic_buffer_2": "list the items each person had. "
        }

        prog_int = EVRAgent(neural_module=NeuralModule(), debug_flag=True)

        prog_int.new_mem_handler(
            program_lines_parent_level=["new_mem(episodic_buffer_0, episodic_buffer_1, episodic_buffer_2)"],
            program_counter_parent_level=0,
            local_variable_dict_parent_level={},
            episodic_buffer_dict_parent_level=episodic_buffer_dict,
            external_textual_buffer_dict=external_chunks
        )


if __name__ == "__main__":
    CheckCartesian.check_cartesian_integrated()
