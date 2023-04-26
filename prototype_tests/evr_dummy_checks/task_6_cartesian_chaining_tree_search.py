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
            if len(re.findall(r'episodic_buffer_0:\s+this is a cartesian chaining tree search task', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 0 can be used to infer the number of items each person had in the beginning', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_3:\s+did john doe have 4 puppies', textual_input)) > 0 and \
                len(textual_list) == 4:

                program_text = \
                    "#0 = 'this is a cartesian task.'; " + \
                    "#1 = 'list the number of items each person had'; " + \
                    "new_mem(#0, episodic_buffer_1, #1); "

            if len(re.findall(r'episodic_buffer_0:\s+this is a cartesian task', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 0 can be used to infer the number of items each person had', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+list the number of items each person had', textual_input)) > 0:

                program_text = \
                    "#2 = ['john smith had 2 toys.', 'john smith had 3 apples.', 'john doe had 2 toys.', 'john doe had 3 apples.']; " + \
                    "add_to_episodic('#0 stores the list the persons.'); " + \
                    "add_to_episodic('#1 stores the list of items.'); " + \
                    "add_to_episodic('#2 stores the number of items each person had.'); "

            if len(re.findall(r'episodic_buffer_0:\s+this is a cartesian task', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 0 can be used to infer the number of items each person had', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+list the number of items each person had', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_3:\s+#0 stores the list the persons', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_4:\s+#1 stores the list of items', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_5:\s+#2 stores the number of items each person had', textual_input)) > 0:

                program_text = \
                    "update_chunk('chunk_0', episodic_buffer_5);" + \
                    "#0 = 'chunk 0 answers how many items each person had in the beginning.';" + \
                    "return(#0);"

            if len(re.findall(r'episodic_buffer_0:\s+this is a cartesian chaining tree search task', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_1:\s+chunk 0 can be used to infer the number of items each person had in the beginning', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_2:\s+chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_3:\s+did john doe have 4 puppies', textual_input)) > 0 and \
                len(re.findall(r'episodic_buffer_4:\s+chunk 0 answers how many items each person had in the beginning', textual_input)) > 0:

                program_text = "clear_mem(); "

            return_var = program_text

        if textual_input.startswith("clear_mem: "):

            return_var = \
                "episodic_buffer_0: 'this is a chaining tree search task.' " + \
                "episodic_buffer_1: 'chunk 0 answers how many items each person had in the beginning.' " + \
                "episodic_buffer_2: 'chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging.' " + \
                "episodic_buffer_3: 'did John Doe have 4 puppies?' "

        return return_var


class CheckCartesianTreeSearch:
    prog_int = EVRAgent(neural_module=NeuralModule())

    @classmethod
    def check_chaining_tree_search_integrated(cls):
        external_chunks = {
            "chunk_0": {
                "statement_0": "each of John Smith and John Doe had 2 toys and 3 apples. ",
            }
        }

        episodic_buffer_dict = {
            "episodic_buffer_0": "this is a cartesian chaining tree search task. ",
            "episodic_buffer_1": "chunk 0 can be used to infer the number of items each person had in the beginning.",
            "episodic_buffer_2": "chunk 1 to chunk 2 can be used to infer how many items each person had after exchanging.",
            "episodic_buffer_3": "did John Doe have 4 puppies?"
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
    CheckCartesianTreeSearch.check_chaining_tree_search_integrated()
