import re

from transformers import T5Tokenizer

from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent
from preliminary_experiments.data_generation.data_2_tree_search_inter_at_once import GenerateTreeSearchTrainingDataAllAtOnce
from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils


class StringExtractionUtils:

    @classmethod
    def extract_beginning_statement(cls, statement):

        pattern = r'.\s+([a-zA-Z\s]+ had \d+ [a-z\s]+ in the beginning)'

        matched = re.findall(pattern, statement)

        if len(matched) > 0:
            return matched[0] + ". "
        else:
            return ""

    @classmethod
    def extract_name_from_beginning_statement(cls, statement):

        pattern = r'.\s+([a-zA-Z\s]+) had \d+ [a-z\s]+ in the beginning'

        matched = re.findall(pattern, statement)

        if len(matched) > 0:
            return matched[0]
        else:
            return ""

    @classmethod
    def extract_item_from_beginning_statement(cls, statement):

        pattern = r'.\s+[a-zA-Z\s]+ had \d+ ([a-z\s]+) in the beginning'

        matched = re.findall(pattern, statement)

        if len(matched) > 0:
            return matched[0]
        else:
            return ""

    @classmethod
    def extract_chunk_num_from_statement(cls, statement):

        pattern = r'chunk \d+'

        matched = re.findall(pattern, statement)

        if len(matched) > 0:
            return matched[0]
        else:
            return ""


class NeuralModule:

    def __init__(self):
        pass

    def inference(self, textual_input):

        return_var = ""
        textual_input = textual_input.lower()
        if textual_input.startswith("generate_program: "):
            textual_input = textual_input.replace("generate_program: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]
            textual_input_cleaned = " ".join(textual_list).lower()

            program_text = ""
            if len(textual_list) == 4 and \
                "this is a chaining tree search task" in textual_input_cleaned and \
                "chunk 0 can be used to infer how many items each person had in the beginning" in textual_input_cleaned and \
                "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging" in textual_input_cleaned and \
                "did roger williams have 7 toy bears" in textual_input_cleaned:

                program_text = \
                    "#0 = 'how many items did each person have after exchanging?'; " + \
                    "new_mem(episodic_buffer_1, episodic_buffer_2, #0); "

            if len(textual_list) == 3 and \
                "chunk 0 can be used to infer how many items each person had in the beginning" in textual_input_cleaned and \
                "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging" in textual_input_cleaned and \
                "how many items did each person have after exchanging" in textual_input_cleaned:

                program_text = \
                    "#0 = []; " + \
                    "#1 = 'this is a chaining task'; " + \
                    "#2 = 'chunk_0'; " + \
                    "while check_next_statement(#2); " + \
	                    "#3 = get_next_statement_num(#2); " + \
	                    "#4 = get_statement(#2, #3); " + \
	                    "#5 = subq(#4, episodic_buffer_2); " + \
	                    "#6 = rewrite(episodic_buffer_0, #4); " + \
	                    "new_mem(#1, episodic_buffer_1, #5, #6); " + \
	                    "#0 = append_to_list(#0, episodic_buffer_3); " + \
	                    "del('episodic_buffer_3'); " + \
                    "end_while; " + \
                    "add_to_episodic('#0 stores the number of items each person had in the beginning.'); "

            if len(textual_list) == 4 and \
                len(re.findall(r'episodic_buffer_0:\s+this is a chaining task', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_1:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_2:\s+how many [a-z\s]+ did [a-zA-Z\s]+ have', textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_3:\s+according to chunk \d+, [a-zA-Z\s]+ had \d+ [a-zA-Z\s]+ in the beginning', textual_input)) > 0:
                if "billy jackson had" in textual_list[3].lower():
                    program_text = "add_to_episodic('billy jackson had 6 peaches. '); "

                if "jonathan gutierrez had " in textual_list[3].lower():
                    program_text = "add_to_episodic('jonathan gutierrez had 1 kitten. ')"

                if "brian allen" in textual_list[3].lower():
                    program_text = "add_to_episodic('brian allen had 3 peaches. ')"

            if len(re.findall(r'episodic_buffer_0:\s+this is a chaining task', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+how many [a-z\s]+ did [a-zA-Z\s]+ have', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_3:\s+according to chunk \d+, [a-zA-Z\s]+ had \d+ [a-zA-Z\s]+ in the beginning', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_4:\s+[a-zA-Z\s]+ had \d+ [a-zA-Z\s]+', textual_input)) > 0:

                program_text = "return(episodic_buffer_4)"

            if len(re.findall(r'episodic_buffer_0:\s+chunk 0 can be used to infer how many items each person had in the beginning', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+how many items did each person have after exchanging', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_3:\s+#0 stores the number of items each person had in the beginning', textual_input)) > 0:

                program_text = \
                    "update_chunk('chunk_0', #0); " + \
                    "#1 = list_chunk_nums('chunk_1', 'chunk_2'); " + \
                    "for #2 in #1; " + \
	                    "del(#2); " + \
                    "end_for; " + \
                    "clean_chunks(); " + \
                    "#3 = 'the task is converted to a tree search task.'; " + \
                    "return(#3); "

            if len(re.findall(r'episodic_buffer_0:\s+this is a chaining tree search task', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 0 can be used to infer how many items each person had in the beginning', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_3:\s+did roger williams have 7 toy bears', textual_input)) > 0 and \
                    len(re.findall(
                        r'episodic_buffer_4:\s+the task is converted to a tree search task',
                        textual_input)) > 0:

                program_text = "clear_mem()"

            # The following conditions below to the chaining task:

            return_var = program_text

        if textual_input.startswith("clear_mem: "):

            if len(re.findall(r'episodic_buffer_0:\s+this is a chaining tree search task', textual_input)) > 0 and \
                    len(re.findall(
                        r'episodic_buffer_1:\s+chunk 0 can be used to infer how many items each person had in the beginning',
                        textual_input)) > 0 and \
                    len(re.findall(
                        r'episodic_buffer_2:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging',
                        textual_input)) > 0 and \
                    len(re.findall(r'episodic_buffer_3:\s+did roger williams have 7 toy bears', textual_input)) > 0 and \
                    len(re.findall(
                        r'episodic_buffer_4:\s+the task is converted to a tree search task',
                        textual_input)) > 0:

                return_var = \
                    "episodic_buffer_0: 'this is a tree search task.' " + \
                    "episodic_buffer_1: 'did roger williams have 7 toy bears?' "

        if textual_input.startswith("subq: "):
            beginning_statement = \
                StringExtractionUtils.extract_beginning_statement(textual_input)

            if "how many items did each person have after exchanging" in textual_input and \
                len(beginning_statement) > 0:

                main_chara = StringExtractionUtils.extract_name_from_beginning_statement(beginning_statement)
                item = StringExtractionUtils.extract_item_from_beginning_statement(beginning_statement)

                return_var = "how many " + item + " did " + main_chara + " have? "

        if textual_input.startswith("rewrite: "):

            if "chunk 0 can be used to infer how many items each person had in the beginning. " in textual_input:
                beginning_statement = \
                    StringExtractionUtils.extract_beginning_statement(textual_input)
                chunk_num = StringExtractionUtils.extract_chunk_num_from_statement(textual_input)
                if len(beginning_statement) > 0 and len(chunk_num) > 0:

                    return_var = "according to " + chunk_num + ", " + beginning_statement

        return return_var


class CheckChainingTreeSearch:
    prog_int = EVRAgent(neural_module=NeuralModule())

    @classmethod
    def check_chaining_tree_search_integrated(cls):
        external_chunks = {
            "chunk_0": {
                "statement_0": "Billy Jackson had 5 peaches in the beginning.",
                "statement_1": "Jonathan Gutierrez had 4 kitten in the beginning.",
                "statement_2": "Brian Allen had 5 peaches in the beginning.",
            },
            "chunk_1": {
                "statement_0": "billy jackson gave x 2 peaches. ",
                "statement_1": "y gave billy jackson 3 peaches. "
            },
            "chunk_2": {
                "statement_0": "jonathan Gutierrez gave x 3 kittens. ",
            },
            "chunk_3": {
                "statement_0": "brian allen gave x 4 peaches. ",
                "statement_1": "y gave brian allen 2 peaches. "
            },
            "chunk_4": {
                "statement_0": "If Alan Nguyen had 2 peaches then Johnny Turner had 7 toy bears.",
                "statement_1": "If Jonathan Gutierrez had 15 kittens then Arthur Phillips had 10 pens.",
                "statement_2": "If Billy Jackson had 6 peaches then Roger Williams had 7 pens."
            },
            "chunk_5": {
                "statement_0": "If Walter Martin had 1 pear and Jonathan Gutierrez had 15 kittens "
                               "then Arthur Phillips had 10 pens.",
                "statement_1": "If Brian Allen had 3 peaches then Roger Williams had 7 toy bears."
            }
        }

        episodic_buffer_dict = {
            "episodic_buffer_0": "this is a chaining tree search task. ",
            "episodic_buffer_1": "chunk 0 can be used to infer how many items each person had in the beginning. ",
            "episodic_buffer_2": "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging. ",
            "episodic_buffer_3": "Did Roger Williams have 7 toy bears? "
        }

        prog_int = EVRAgent(neural_module=NeuralModule(), debug_flag=True)

        prog_int.new_mem_handler(
            program_lines_parent_level=[
                "new_mem(episodic_buffer_0, episodic_buffer_1, episodic_buffer_2, episodic_buffer_3)"],
            program_counter_parent_level=0,
            local_variable_dict_parent_level={},
            episodic_buffer_dict_parent_level=episodic_buffer_dict,
            external_textual_buffer_dict=external_chunks
        )


if __name__ == "__main__":
    CheckChainingTreeSearch.check_chaining_tree_search_integrated()
