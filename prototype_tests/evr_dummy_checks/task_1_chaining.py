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
        textual_input = textual_input.lower()
        if textual_input.startswith("generate_program"):

            textual_input = textual_input.replace("generate_program: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]

            textual_input_cleaned = " ".join(textual_list).lower()

            program_text = ""
            if len(textual_list) == 4 and \
                    "this is a chaining task" in textual_input_cleaned and \
                    "chunk 0 answers how many items each person had in the beginning" in textual_input_cleaned and \
                    "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging" in textual_input_cleaned and \
                    len(re.findall(r'episodic_buffer_3:\s+how many toys did john doe have', textual_input)) > 0:
                program_text = \
                    "#0 = 'according to chunk 0, how many toys did john doe have in the beginning'; " + \
                    "new_mem(#0);"

            if len(textual_list) == 1 and \
                len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, how many toys did john doe have in the beginning', textual_input)) > 0:

                program_text = \
                    "#0 = 'chunk_0'; " + \
                    "while check_next_statement(#0); " + \
                    "#1 = get_next_statement_num(#0); " + \
                    "#2 = get_statement(#0, #1); " + \
                    "#3 = qa(#2, episodic_buffer_0); " + \
                    "if #3 != 'None'; " + \
                    "#4 = #3; " + \
                    "else; " + \
                    "pass; " + \
                    "end_if; " + \
                    "end_while; " + \
                    "add_to_episodic(#4); "

            if len(textual_list) == 2 and \
                len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, how many toys did john doe have in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0:

                program_text = "return(episodic_buffer_1);"

            if len(textual_list) == 5 and \
                    "this is a chaining task" in textual_input_cleaned and \
                    "chunk 0 answers how many items each person had in the beginning" in textual_input_cleaned and \
                    "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging" in textual_input_cleaned and \
                    len(re.findall(r'episodic_buffer_3:\s+how many toys did john doe have', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_4:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0:
                program_text = "clear_mem();"

            if len(textual_list) == 4 and \
                    "this is a chaining task" in textual_input_cleaned and \
                    "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging" in textual_input_cleaned and \
                    len(re.findall(r'episodic_buffer_2:\s+how many toys did john doe have', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_3:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0:

                program_text = \
                    "#0 = 'according to the chunks from chunk 1 to chunk 2, how many toys did john doe have after exchanging'; " + \
                    "new_mem(episodic_buffer_3, #0); "

            if len(textual_list) == 2 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+according to the chunks from chunk 1 to chunk 2, how many toys did john doe have after exchanging', textual_input)) > 0:

                program_text = \
                    "#0 = 'according to the chunks from chunk 1 to chunk 2, which chunk can be used to infer how many toys john doe had after exchanging'; " + \
                    "new_mem(#0); "

            if len(textual_list) == 1 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to the chunks from chunk 1 to chunk 2, which chunk can be used to infer how many toys john doe had after exchanging', textual_input)) > 0:

                program_text = \
                    "#0 = list_chunk_nums('chunk_1', 'chunk_2'); " + \
                    "for #1 in #0; " + \
                    "#2 = get_chunk(#1); " + \
                    "#3 = 'can this chunk be used to infer how many toys john doe had after exchanging'; " + \
                    "#4 = qa(#2, #3); " + \
                    "if #4 == 'True'; " + \
                    "#5 = #1; " + \
                    "else; " + \
                    "pass; " + \
                    "end_if; " + \
                    "end_for; " + \
                    "#6 = rewrite(episodic_buffer_0, #5); " + \
                    "add_to_episodic(#6);"

            if len(textual_list) == 2 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to the chunks from chunk 1 to chunk 2, which chunk can be used to infer how many toys john doe had after exchanging', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+chunk \d+ can be used to infer how many toys john doe had after exchanging', textual_input)) > 0:

                program_text = "return(episodic_buffer_1);"

            if len(textual_list) == 3 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+according to the chunks from chunk 1 to chunk 2, how many toys did john doe have after exchanging', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_2:\s+chunk \d+ can be used to infer how many toys john doe had after exchanging', textual_input)) > 0:
                program_text = \
                    "#0 = 'according to chunk 2, how many toys did john doe have after exchanging?'; " + \
                    "new_mem(episodic_buffer_0, #0); "

            if len(textual_list) == 2 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+according to chunk 2, how many toys did john doe have after exchanging', textual_input)) > 0:
                program_text = \
                    "#0 = 'chunk_2'; " + \
                    "#1 = 'John Doe had 2 toys. '; " + \
                    "while check_next_statement(#0); " + \
                    "#2 = get_next_statement_num(#0); " + \
                    "#3 = get_statement(#0, #2); " + \
                    "#4 = 'how many toys did John Doe have after exchanging?'; " + \
                    "#1 = qa(#1, #3, #4); " + \
                    "end_while; " + \
                    "add_to_episodic(#1); "

            if len(textual_list) == 3 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+according to chunk \d+, how many toys did john doe have after exchanging', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_2:\s+john doe had \d+ toys', textual_input)) > 0:
                program_text = \
                    "#0 = rewrite(episodic_buffer_1, episodic_buffer_2); " + \
                    "return(#0); "

            if len(textual_list) == 4 and \
                    len(re.findall(r'episodic_buffer_0:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+according to the chunks from chunk 1 to chunk 2, how many toys did john doe have after exchanging', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_2:\s+chunk \d+ can be used to infer how many toys john doe had after exchanging', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_3:\s+john doe had \d+ toys after exchanging', textual_input)) > 0:
                program_text = "return(episodic_buffer_3); "

            if len(textual_list) == 5 and \
                    "this is a chaining task" in textual_input_cleaned and \
                    "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging" in textual_input_cleaned and \
                    len(re.findall(r'episodic_buffer_2:\s+how many toys did john doe have', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_3:\s+according to chunk 0, john doe had \d+ toys in the beginning', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_4:\s+john doe had \d+ toys after exchanging', textual_input)) > 0:
                program_text = \
                    "#0 = rewrite(episodic_buffer_2, episodic_buffer_4); " + \
                    "return(#0); "

            return_var = program_text

        if textual_input.startswith("qa:"):

            if "according to chunk 0, how many toys did john doe have in the beginning" in textual_input.lower():
                num = re.findall(r'john doe had (\d+) toys in the beginning', textual_input.lower())

                if len(num) > 0:
                    num = num[0]
                    return_var = "according to chunk 0, John Doe had " + num + " toys in the beginning. "

                else:
                    return_var = "None"

            if "can this chunk be used to infer how many toys john doe had after exchanging" in textual_input.lower():
                if "gave john doe" in textual_input or "john doe gave" in textual_input:
                    return_var = "True"
                else:
                    return_var = "False"

            if "how many toys did john doe have after exchanging?" in textual_input.lower() and \
                    ("john doe gave" in textual_input.lower() or "gave john doe" in textual_input.lower()):

                start_num = re.findall(r'john doe had (\d+) toys', textual_input.lower())

                neg_num = re.findall(r'john doe gave .+ (\d+) toys', textual_input.lower())

                pos_num = re.findall(r'gave john doe (\d+) toys', textual_input.lower())

                if len(pos_num) > 0:
                    end_num = int(start_num[0]) + int(pos_num[0])

                elif len(neg_num) > 0:
                    end_num = int(start_num[0]) - int(neg_num[0])

                else:
                    end_num = int(start_num[0])

                return_var = "john doe had " + str(end_num) + " toys. "

        if textual_input.startswith("rewrite: "):
            if "which chunk can be used to infer how many toys john doe" in textual_input.lower():
                chunk_num = re.findall(r'chunk_(\d+)', textual_input)[0]
                return_var = "chunk " + chunk_num + " can be used to infer how many toys John Doe had after exchanging"

            if "according to chunk 2, how many toys did john doe have after exchanging?" in textual_input.lower() and \
                "john doe had 3 toys" in textual_input.lower():

                return_var = "john doe had 3 toys after exchanging. "

            if "john doe had 3 toys after exchanging" in textual_input.lower():
                return_var = "3"

        if textual_input.startswith("clear_mem: "):
            return_var = \
                "episodic_buffer_0: 'this is a chaining task.' " + \
                "episodic_buffer_1: 'chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging.' " + \
                "episodic_buffer_2: 'how many toys did john doe have.' " + \
                "episodic_buffer_3: 'according to chunk 0, john doe had 2 toys in the beginning.' "

        return return_var


class CheckChaining:

    prog_int = EVRAgent(neural_module=NeuralModule())

    @classmethod
    def check_task1_snippets(cls):

        cases = [
            {
                "lines": [
                    "while check_next_chunk()",
                        "#1 = get_next_chunk_num()",
                        "#2 = get_chunk(#1)",
                        "#3 = 'Can this chunk be used to answer how many toys A has?'",
                        "#4 = qa(#2, #3)",
                        "if #4 == True",
                            "#5 = #1",
                        "else",
                            "pass",
                        "end_if",
                    "end_while",
                    "#6 = rewrite(episodic_buffer_0, #5)"
                ],
                "lvd": {},
                "ebd": {"episodic_buffer_0": "which chunk can be used to answer how many toys A has? "},
                "emd": {"chunk_0": "A has 12 apples", "chunk_1": "A has 12 toys"}
            },

            {
                "lines": [
                    "#1 = 10",
                    "#0 = 'chunk_0'",
                    "while check_next_statement(#0)",
	                    "#2 = get_next_statement_num(#0)",
                        "#3 = get_statement(#0, #2)",
	                    "#4 = 'how many toys did other give A?'",
                        "#5 = qa(#3, #4)",
                        "#1 = arith_sum(#1, #5)",
	                "end_while",
                    "#6 = rewrite(episodic_buffer_1, #1)",
                ],
                "lvd": {},
                "ebd": {"episodic_buffer_0": "which chunk can be used to answer how many toys A has?",
                        "episodic_buffer_1": "how many toys does A have after A exchange toys with others?"},
                "emd": {
                    "chunk_0": {
                        "statement1": "A gives B 3 toys. ",
                        "statement2": "C gives A 2 toys. ",
                    },
                    "chunk_1": "A has 12 toys"
                }
            },
        ]

        for case in cases:
            print("=" * 40)
            print("lines:", case["lines"])
            print("lvd:", case["lvd"])
            print("ebd:", case["ebd"])
            print("emd:", case["emd"])
            print("-" * 40)

            case["lvd"] = cls.prog_int.program_handler(case["lines"], case["lvd"], case["ebd"], case["emd"])

            print("lvd:", case["lvd"])

            print("=" * 40)

    @classmethod
    def check_chaining_integrated(cls):

        external_chunks = {
            "chunk_0": {
                "statement_0": "james smith had 3 bears in the beginning. ",
                "statement_1": "john doe had 2 toys in the beginning. "
            },
            "chunk_1": {
                "statement_0": "michael david gave james smith 2 bears. ",
                "statement_1": "richard david gave james smith 2 bears. "
            },
            "chunk_2": {
                "statement_0": "michael david gave john doe 3 toys. ",
                "statement_1": "john doe gave richard david 2 toys. ",
            },
        }

        episodic_buffer_dict = {
            "episodic_buffer_0": "this is a chaining task. ",
            "episodic_buffer_1": "chunk 0 answers how many items each person had in the beginning. ",
            "episodic_buffer_2": "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging. ",
            "episodic_buffer_3": "how many toys did john doe have. "
        }

        prog_int = EVRAgent(neural_module=NeuralModule(),
                            debug_flag=True)

        prog_int.new_mem_handler(
            program_lines_parent_level=["new_mem(episodic_buffer_0, episodic_buffer_1, episodic_buffer_2, episodic_buffer_3)"],
            program_counter_parent_level=0,
            local_variable_dict_parent_level={},
            episodic_buffer_dict_parent_level=episodic_buffer_dict,
            external_textual_buffer_dict=external_chunks
        )


if __name__ == "__main__":
    #CheckTask1.check_task1_snippets()
    CheckChaining.check_chaining_integrated()
