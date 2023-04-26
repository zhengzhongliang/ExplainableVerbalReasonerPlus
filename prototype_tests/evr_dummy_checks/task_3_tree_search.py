import re

from transformers import T5Tokenizer

from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent
from preliminary_experiments.data_generation.data_2_tree_search_inter_at_once import GenerateTreeSearchTrainingDataAllAtOnce
from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils


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
            if len(textual_list) == 2 and \
                "this is a tree search task." in textual_input_cleaned and \
                "did roger williams have 7 toy bears?" in textual_input_cleaned:
                program_text = "#1 = subq(); new_mem(#1);"

            if len(textual_list) == 1 and \
                "which chunk can prove roger williams had 7 toy bears?" in textual_input_cleaned:
                program_text = "while check_next_chunk(); " + \
                        "#1 = get_next_chunk_num(); " + \
                        "#2 = 'can this chunk prove roger williams had 7 toy bears?'; " + \
                        "#3 = rewrite(#2, #1); " + \
                        "new_mem(#3); " + \
                        "end_while; "

            if len(textual_list) == 1 and \
                "can chunk 0 prove roger williams had 7 toy bears?" in textual_input_cleaned:
                program_text = "#1 = get_chunk('chunk_0'); " + \
                        "#2 = qa(#1, episodic_buffer_0); " + \
                        "add_to_episodic(#2); "

            if len(textual_list) == 2 and \
                "can chunk 0 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "chunk 0 could not prove roger williams had 7 toy bears." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 1 and \
                "can chunk 1 prove roger williams had 7 toy bears?" in textual_input_cleaned:
                program_text = "#1 = get_chunk('chunk_1'); " + \
                        "#2 = qa(#1, episodic_buffer_0); " + \
                        "add_to_episodic(#2); "

            if len(textual_list) == 2 and \
                "can chunk 1 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "chunk 1 could not prove roger williams had 7 toy bears." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 1 and \
                "can chunk 2 prove roger williams had 7 toy bears?" in textual_input_cleaned:
                program_text = "#1 = get_chunk('chunk_2'); " + \
                               "#2 = qa(#1, episodic_buffer_0); " + \
                               "add_to_episodic(#2); "

            if len(textual_list) == 2 and \
                "can chunk 2 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "in order to prove roger williams had 7 toy bears, i need to prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "#1 = subqs(); " + \
                    "for #2 in #1; " + \
                    "new_mem(#2); " + \
                    "end_for; "

            # Starting from this block, this is the second depth search
            if len(textual_list) == 1 and \
                "did brian allen have 3 peaches?" in textual_input_cleaned:
                program_text = "#1 = subq(); new_mem(#1); "

            if len(textual_list) == 1 and \
                "which chunk can prove brian allen had 3 peaches?" in textual_input_cleaned:
                program_text = "while check_next_chunk(); " + \
                        "#1 = get_next_chunk_num(); " + \
                        "#2 = 'can this chunk prove brian allen had 3 peaches?'; " + \
                        "#3 = rewrite(#2, #1); " + \
                        "new_mem(#3); " + \
                        "end_while; "

            if len(textual_list) == 1 and \
                "can chunk 0 prove brian allen had 3 peaches?" in textual_input_cleaned:
                program_text = "#1 = get_chunk('chunk_0'); " + \
                        "#2 = qa(#1, episodic_buffer_0); " + \
                        "add_to_episodic(#2); "

            if len(textual_list) == 2 and \
                "can chunk 0 prove brian allen had 3 peaches?" in textual_input_cleaned and \
                "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 1 and \
                "can chunk 1 prove brian allen had 3 peaches?" in textual_input_cleaned:
                program_text = "#1 = get_chunk('chunk_1'); " + \
                        "#2 = qa(#1, episodic_buffer_0); " + \
                        "add_to_episodic(#2); "

            if len(textual_list) == 2 and \
                "can chunk 1 prove brian allen had 3 peaches?" in textual_input_cleaned and \
                "chunk 1 could not prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 1 and \
                "can chunk 2 prove brian allen had 3 peaches?" in textual_input_cleaned:
                program_text = "#1 = get_chunk('chunk_2'); " + \
                               "#2 = qa(#1, episodic_buffer_0); " + \
                               "add_to_episodic(#2); "

            if len(textual_list) == 2 and \
                "can chunk 2 prove brian allen had 3 peaches?" in textual_input_cleaned and \
                "chunk 2 could not prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 4 and \
                "which chunk can prove brian allen had 3 peaches?" in textual_input_cleaned and \
                "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned and \
                "chunk 1 could not prove brian allen had 3 peaches." in textual_input_cleaned and \
                "chunk 2 could not prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "clear_mem(); "

            if len(textual_list) == 2 and \
                "which chunk can prove brian allen had 3 peaches?" in textual_input_cleaned and \
                    "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 2 and \
                "did brian allen have 3 peaches?" in textual_input_cleaned and \
                    "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 3 and \
                "can chunk 2 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "in order to prove roger williams had 7 toy bears, i need to prove brian allen had 3 peaches." in textual_input_cleaned and \
                    "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned:
                program_text = "clear_mem(); "
                # After clear memory, it should return "chunk 2 can prove roger williams had 7 toy bears"

            if len(textual_list) == 2 and \
                "can chunk 2 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "chunk 2 can prove roger williams had 7 toy bears." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 4 and \
                "which chunk can prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "chunk 0 could not prove roger williams had 7 toy bears." in textual_input_cleaned and \
                "chunk 1 could not prove roger williams had 7 toy bears." in textual_input_cleaned and \
                "chunk 2 can prove roger williams had 7 toy bears." in textual_input_cleaned:
                program_text = "clear_mem(); "

            if len(textual_list) == 2 and \
                "which chunk can prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                "chunk 2 can prove roger williams had 7 toy bears." in textual_input_cleaned:
                program_text = "return(episodic_buffer_1); "

            if len(textual_list) == 3 and \
                "this is a tree search task." in textual_input_cleaned and \
                "did roger williams have 7 toy bears?" in textual_input_cleaned and \
                "chunk 2 can prove roger williams had 7 toy bears." in textual_input_cleaned:

                program_text = \
                    "#0 = rewrite(episodic_buffer_1, episodic_buffer_2); " + \
                    "return(#0); "

            return_var = program_text

        if textual_input.startswith("qa: "):
            # The QA module is used for 6 times. 3 times when checking the 3 chunks for
            # the roger williams query, 3 times when checking the 3 chunks for the
            # brian allen chunks.

            if "billy jackson had 6 peaches." in textual_input and \
                "jonathan gutierrez had 1 kitten." in textual_input and \
                "brian allen had 3 peaches." in textual_input:
                if "can chunk 0 prove roger williams had 7 toy bears?" in textual_input:
                    return_var = "chunk 0 could not prove roger williams had 7 toy bears."

                if "can chunk 0 prove brian allen had 3 peaches?" in textual_input:
                    return_var = "chunk 0 can prove brian allen had 3 peaches."

            if "if alan nguyen had 2 peaches then johnny turner had 7 toy bears." in textual_input and \
                "if jonathan gutierrez had 15 kittens then arthur phillips had 10 pens." in textual_input and \
                "if billy jackson had 6 peaches then roger williams had 7 pens." in textual_input:
                if "can chunk 1 prove roger williams had 7 toy bears?" in textual_input:
                    return_var = "chunk 1 could not prove roger williams had 7 toy bears."

                if "can chunk 1 prove brian allen had 3 peaches?" in textual_input:
                    return_var = "chunk 1 could not prove brian allen had 3 peaches."

            if "if walter martin had 1 pear and jonathan gutierrez had 15 kittens then arthur phillips had 10 pens." \
                in textual_input and \
                "if brian allen had 3 peaches then roger williams had 7 toy bears." in textual_input:
                if "can chunk 2 prove roger williams had 7 toy bears?" in textual_input:
                    return_var = "in order to prove roger williams had 7 toy bears, i need to prove brian allen had 3 peaches."

                if "can chunk 2 prove brian allen had 3 peaches?" in textual_input:
                    return_var = "chunk 2 could not prove brian allen had 3 peaches."

        if textual_input.startswith("rewrite: "):

            number_extract_pattern = r'chunk_(\d+)'

            if "can this chunk prove roger williams had 7 toy bears?" in textual_input:
                chunk_num = re.findall(number_extract_pattern, textual_input)[0]
                return_var = "can chunk " + chunk_num + " prove roger williams had 7 toy bears?"

            if "can this chunk prove brian allen had 3 peaches?" in textual_input:
                chunk_num = re.findall(number_extract_pattern, textual_input)[0]
                return_var = "can chunk " + chunk_num + " prove brian allen had 3 peaches?"

            if "did roger williams have 7 toy bears?" in textual_input.lower() and \
                "chunk 2 can prove roger williams had 7 toy bears." in textual_input.lower():

                return "True"

        if textual_input.startswith("subq: "):
            textual_input = textual_input.replace("subq: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]

            textual_input_cleaned = " ".join(textual_list).lower()

            if len(textual_list) == 2 and \
                "this is a tree search task." in textual_input_cleaned and \
                "did roger williams have 7 toy bears?" in textual_input_cleaned:
                return_var = "which chunk can prove roger williams had 7 toy bears?"

            if len(textual_list) == 1 and \
                "did brian allen have 3 peaches?" in textual_input_cleaned:
                return_var = "which chunk can prove brian allen had 3 peaches?"

        if textual_input.startswith("subqs: "):
            textual_input = textual_input.replace("subqs: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]

            textual_input_cleaned = " ".join(textual_list).lower()

            if len(textual_list) == 2 and \
                    "can chunk 2 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                    "i need to prove brian allen had 3 peaches." in textual_input_cleaned:
                return_var = "['did brian allen have 3 peaches?']"

        if textual_input.startswith("clear_mem: "):
            textual_input = textual_input.replace("clear_mem: ", "")
            textual_list = [x[1:] for x in textual_input.split("episodic_buffer_")]
            textual_list = [x for x in textual_list if x != ""]

            textual_input_cleaned = " ".join(textual_list).lower()

            if len(textual_list) == 4 and \
                    "which chunk can prove brian allen had 3 peaches?" in textual_input_cleaned and \
                    "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned and \
                    "chunk 1 could not prove brian allen had 3 peaches." in textual_input_cleaned and \
                    "chunk 2 could not prove brian allen had 3 peaches." in textual_input_cleaned:
                return_var = "episodic_buffer_0: 'which chunk can prove brian allen had 3 peaches?' " + \
                             "episodic_buffer_1: 'chunk 0 can prove brian allen had 3 peaches.' "

            if len(textual_list) == 3 and \
                    "can chunk 2 prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                    "in order to prove roger williams had 7 toy bears, i need to prove brian allen had 3 peaches." in textual_input_cleaned and \
                    "chunk 0 can prove brian allen had 3 peaches." in textual_input_cleaned:
                return_var = "episodic_buffer_0: 'can chunk 2 prove roger williams had 7 toy bears?' " + \
                             "episodic_buffer_1: 'chunk 2 can prove roger williams had 7 toy bears.' "

            if len(textual_list) == 4 and \
                    "which chunk can prove roger williams had 7 toy bears?" in textual_input_cleaned and \
                    "chunk 0 could not prove roger williams had 7 toy bears." in textual_input_cleaned and \
                    "chunk 1 could not prove roger williams had 7 toy bears." in textual_input_cleaned and \
                    "chunk 2 can prove roger williams had 7 toy bears." in textual_input_cleaned:
                return_var = "episodic_buffer_0: 'which chunk can prove roger williams had 7 toy bears?' " + \
                             "episodic_buffer_1: 'chunk 2 can prove roger williams had 7 toy bears.' "

        return return_var


class CheckTreeSearch:

    @classmethod
    def load_and_check_examples(cls, machine_switch="mac"):

        instances_all_depth = ExpDatasetUtils.load_data(seed=0,
                                                        n_train=20000,
                                                        machine_switch=machine_switch,
                                                        data_pattern="tree_search_v0.5",
                                                        dev_ratio=0.1)

        tokenizer = T5Tokenizer.from_pretrained("allenai/unifiedqa-t5-large")

        for instance in instances_all_depth[2]["train"][18000: 18100]:
            target_text, target_text_len, traversal_history, depth_history = \
                GenerateTreeSearchTrainingDataAllAtOnce.generate_tree_search_backward_select_evidence_one_instance(
                    instance, tokenizer, debug_flag=True
                )

            print("-" * 40)
            print(target_text)
            print(target_text_len)
            print(traversal_history)
            print(depth_history)
            input("--------")

        '''
        Example:
        question: Did Roger Williams have 7 toy bears?
        pred: True  answer: Yes

        Context:
        0: Billy Jackson had 6 peaches.
        1: Jonathan Gutierrez had 1 kitten.
        2: Brian Allen had 3 peaches.
        3: If Alan Nguyen had 2 peaches then Johnny Turner had 7 toy bears.
        4: If Jonathan Gutierrez had 15 kittens then Arthur Phillips had 10 pens.
        5: If Billy Jackson had 6 peaches then Roger Williams had 7 pens.
        6: If Walter Martin had 1 pear and Jonathan Gutierrez had 15 kittens then Arthur Phillips had 10 pens.
        7: If Brian Allen had 3 peaches then Roger Williams had 7 toy bears.

        Traversal history:
        7: If Brian Allen had 3 peaches then Roger Williams had 7 toy bears.
            2: Brian Allen had 3 peaches.
        '''

    @classmethod
    def check_tree_search_integrated(cls):

        external_chunks = {
            "chunk_0": {
                "statement_0": "Billy Jackson had 6 peaches.",
                "statement_1": "Jonathan Gutierrez had 1 kitten.",
                "statement_2": "Brian Allen had 3 peaches.",
            },
            "chunk_1": {
                "statement_0": "If Alan Nguyen had 2 peaches then Johnny Turner had 7 toy bears.",
                "statement_1": "If Jonathan Gutierrez had 15 kittens then Arthur Phillips had 10 pens.",
                "statement_2": "If Billy Jackson had 6 peaches then Roger Williams had 7 pens."
            },
            "chunk_2": {
                "statement_0": "If Walter Martin had 1 pear and Jonathan Gutierrez had 15 kittens "
                               "then Arthur Phillips had 10 pens.",
                "statement_1": "If Brian Allen had 3 peaches then Roger Williams had 7 toy bears."
            }
        }

        episodic_buffer_dict = {
            "episodic_buffer_0": "this is a tree search task. ",
            "episodic_buffer_1": "Did Roger Williams have 7 toy bears?",
        }

        evr_agent = EVRAgent(neural_module=NeuralModule(),
                             debug_flag=True)

        evr_agent.new_mem_handler(
            program_lines_parent_level=["new_mem(episodic_buffer_0, episodic_buffer_1)"],
            program_counter_parent_level=0,
            local_variable_dict_parent_level={},
            episodic_buffer_dict_parent_level=episodic_buffer_dict,
            external_textual_buffer_dict=external_chunks
        )


if __name__ == "__main__":
    #CheckTask2.load_and_check_examples()
    CheckTreeSearch.check_tree_search_integrated()
