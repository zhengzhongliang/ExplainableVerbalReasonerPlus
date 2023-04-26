import re
import math

from preliminary_experiments.experiments_evr.evr_class.utils_evr_arithmetic import ArithmeticOperators
from preliminary_experiments.experiments_evr.evr_class.utils_evr_string_processing import StringProcessor
from preliminary_experiments.experiments_evr.evr_class.utils_evr_debug import UtilsEVRDebug


class EVRAgent:

    """
    Design of the chunk/fact/rule buffer:
    buffer_memory = {
        chunk_0: {
            statement_0: {

            },
            ...
        }
        ...
    }

    The hierarchical structure of the handlers:
     - key word handler:
         - recursion handler
            - (function) new_mem
            - (function) return
         - if, while, for handler
         - episodic buffer handler
            - (function) add_to_episodic
         - assignment handler
            - (function) rewrite
            - (function) subq
            - (function) qa
            - (function) chunk
            - (function) statement
            - (function) arith_sum

    """

    def __init__(self, neural_module, debug_flag=False, print_chunk=True):

        self.default_variables = {
            "None": None,
            "True": True,
            "False": False
        }

        self.arithmetic_operators = ArithmeticOperators.get_arithmetic_operators()

        self.keywords = {
            "while", "end_while", "if", "else", "end_if", "for", "end_for"
        }

        self.keyword_handlers = {
            "assign": self.assign_handler,
            "while": self.while_handler,
            "if": self.if_handler,
            "for": self.for_handler,
            "pass": self.pass_handler,
            "add_to_episodic": self.add_to_episodic_handler,
            "clear_mem": self.clear_mem_handler,
            "new_mem": self.new_mem_handler,
            "return": self.return_handler,
            "del": self.del_handler,
            "update_chunk": self.update_chunk_handler,
            "clean_chunks": self.clean_chunks_handler,
        }

        self.function_handlers = {
            "qa": self.qa_handler,
            "rewrite": self.rewrite_handler,
            "subq": self.subq_handler,
            "subqs": self.subqs_handler,
            "check_next_chunk": self.check_next_chunk_handler,
            "get_next_chunk_num": self.get_next_chunk_num_handler,
            "get_chunk": self.get_chunk_handler,
            "check_next_statement": self.check_next_statement_handler,
            "get_next_statement_num": self.get_next_statement_num_handler,
            "get_statement": self.get_statement_handler,
            "arith_sum": self.arith_sum_handler,
            "append_to_list": self.append_to_list_handler,
            "list_chunk_nums": self.list_chunk_nums_handler,
        }

        self.neural_module = neural_module

        self.debug_flag = debug_flag
        self.print_chunk = print_chunk
        self.indent_level = -1   # This field is used to print the debugging information nicely

    def get_variable_value(self, var_str, local_variable_dict, episodic_buffer_dict):

        """
        var str could either be a variable name or a constant, like a string, a number, or some default value.
        :param var_str:
        :param local_variable_dict:
        :return:
        """

        var_str = StringProcessor.remove_string_leading_spaces(var_str)

        if var_str.startswith("[") and var_str.endswith("]"):

            list_var_strs = StringProcessor.process_list(var_str)

            list_vars = [self.get_variable_value(v, local_variable_dict, episodic_buffer_dict) for v in list_var_strs]
            return list_vars

        elif var_str.startswith("'") and var_str.endswith("'"):   # var is a string
            return var_str[1: -1]

        elif var_str.lstrip("-").isdigit():   # var is a digit
            return int(var_str)

        elif var_str in self.default_variables:   # var is a default variable
            return self.default_variables[var_str]

        elif var_str in local_variable_dict:
            return local_variable_dict[var_str]

        elif var_str in episodic_buffer_dict:
            return episodic_buffer_dict[var_str]

        else:
            assert False, "variable value not found!"

    def pass_handler(self,
                     program_lines,
                     program_counter,
                     local_variable_dict,
                     episodic_buffer_dict,
                     external_textual_buffer_dict):

        return program_counter + 1, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def assign_handler(self,
                       program_lines,
                       program_counter,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):
        """
        This function handles each line of code, excluding while, for, if.
        :return:
        """

        current_line = program_lines[program_counter]

        # TODO: do we need assert here?

        var_to_be_assgined, assignment_code = current_line.split("=")
        var_to_be_assgined = var_to_be_assgined.replace(" ", "")

        function_name = StringProcessor.extract_function_name_from_string(assignment_code)

        # Handle the right side is a constant or a variable
        if function_name is None:
            local_variable_dict[var_to_be_assgined] = self.get_variable_value(assignment_code, local_variable_dict, episodic_buffer_dict)
        else:
            func_input_arg_list = StringProcessor.extract_function_args_from_string(assignment_code)

            local_variable_dict[var_to_be_assgined] = self.function_handlers[function_name](func_input_arg_list,
                                                                        local_variable_dict,
                                                                        episodic_buffer_dict,
                                                                        external_textual_buffer_dict)

        program_counter += 1
        return program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def qa_handler(self,
                   func_input_arg_list,
                   local_variable_dict,
                   episodic_buffer_dict,
                   external_textual_buffer_dict):

        var_strs = [
            StringProcessor.form_textual_input_from_chunk_and_statements(
                self.get_variable_value(arg, local_variable_dict, episodic_buffer_dict)
            )
            for arg in func_input_arg_list
        ]

        neural_module_input = "qa: " + " ".join(var_strs)

        return self.neural_module.inference(neural_module_input)

    def rewrite_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        arg_converted_to_strings = [
            StringProcessor.form_textual_input_from_chunk_and_statements(
                self.get_variable_value(arg, local_variable_dict, episodic_buffer_dict)
            )
            for arg in func_input_arg_list
        ]

        arg_converted_to_strings = [
            StringProcessor.remove_string_leading_spaces(x) for x in arg_converted_to_strings
        ]

        arg_converted_to_strings = [
            x + "." if not x.endswith(".") and not x.endswith("?") else x for x in arg_converted_to_strings
        ]

        return self.neural_module.inference("rewrite: " + " ".join(arg_converted_to_strings))

    def subq_handler(self,
                     func_input_arg_list,
                     local_variable_dict,
                     episodic_buffer_dict,
                     external_textual_buffer_dict):

        if len(func_input_arg_list) == 0:
            input_text = " ".join(str(k) + ": " + str(v) for k, v in episodic_buffer_dict.items())
        else:

            arg_converted_to_strings = [
                StringProcessor.form_textual_input_from_chunk_and_statements(
                    self.get_variable_value(arg, local_variable_dict, episodic_buffer_dict)
                )
                for arg in func_input_arg_list
            ]

            arg_converted_to_strings = [
                StringProcessor.remove_string_leading_spaces(x) for x in arg_converted_to_strings
            ]

            arg_converted_to_strings = [
                x + "." if not x.endswith(".") and not x.endswith("?") else x for x in arg_converted_to_strings
            ]

            input_text = " ".join(arg_converted_to_strings)

        return self.neural_module.inference("subq: " + input_text)

    def subqs_handler(self,
                      func_input_arg_list,
                      local_variable_dict,
                      episodic_buffer_dict,
                      external_textual_buffer_dict):

        input_text = " ".join(str(k) + ": " + str(v) for k, v in episodic_buffer_dict.items())

        subqs_str = self.neural_module.inference("subqs: " + input_text)

        subqs_list = self.get_variable_value(subqs_str, local_variable_dict, episodic_buffer_dict)

        return subqs_list

    def check_next_chunk_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        if "_chunk_counter" not in local_variable_dict:
            if len(external_textual_buffer_dict) > 0:
                return True
            else:
                return False

        else:
            if local_variable_dict["_chunk_counter"] < len(external_textual_buffer_dict) - 1:
                return True
            else:
                return False

    def get_next_chunk_num_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        if "_chunk_counter" not in local_variable_dict:
            if len(external_textual_buffer_dict) > 0:
                local_variable_dict["_chunk_counter"] = 0
                chunk_num_raw = list(external_textual_buffer_dict.keys())[local_variable_dict["_chunk_counter"]]
                return chunk_num_raw if isinstance(chunk_num_raw, str) else chunk_num_raw
            else:
                return None

        else:
            if local_variable_dict["_chunk_counter"] < len(external_textual_buffer_dict) - 1:
                local_variable_dict["_chunk_counter"] += 1
                chunk_num_raw = list(external_textual_buffer_dict.keys())[local_variable_dict["_chunk_counter"]]
                return chunk_num_raw if isinstance(chunk_num_raw, str) else chunk_num_raw
            else:
                del local_variable_dict["_chunk_counter"]
                return None

    def get_chunk_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        chunk_num = self.get_variable_value(func_input_arg_list[0], local_variable_dict, episodic_buffer_dict)

        return external_textual_buffer_dict[chunk_num]

    def check_next_statement_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        chunk_num = self.get_variable_value(func_input_arg_list[0], local_variable_dict, episodic_buffer_dict)
        if chunk_num not in external_textual_buffer_dict:
            return False

        chunk_dict = external_textual_buffer_dict[chunk_num]

        if "_" + str(chunk_num) + "_statement_counter" not in local_variable_dict:
            if len(chunk_dict) > 0:
                return True
            else:
                return False

        else:
            if local_variable_dict["_" + str(chunk_num) + "_statement_counter"] < len(chunk_dict) - 1:
                return True
            else:
                return False

    def get_next_statement_num_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        chunk_num = self.get_variable_value(func_input_arg_list[0], local_variable_dict, episodic_buffer_dict)

        if chunk_num not in external_textual_buffer_dict:
            return None

        chunk_dict = external_textual_buffer_dict[chunk_num]

        if "_" + str(chunk_num) + "_statement_counter" not in local_variable_dict:
            if len(chunk_dict) > 0:
                local_variable_dict["_" + str(chunk_num) + "_statement_counter"] = 0
                statement_num_raw = list(chunk_dict.keys())[local_variable_dict["_" + str(chunk_num) + "_statement_counter"]]

                return statement_num_raw
            else:
                return None

        else:
            if local_variable_dict["_" + str(chunk_num) + "_statement_counter"] < len(chunk_dict) - 1:
                local_variable_dict["_" + str(chunk_num) + "_statement_counter"] += 1
                statement_num_raw = list(chunk_dict.keys())[
                    local_variable_dict["_" + str(chunk_num) + "_statement_counter"]]
                return statement_num_raw
            else:
                del local_variable_dict["_" + str(chunk_num) + "_statement_counter"]
                return None

    def get_statement_handler(self,
                       func_input_arg_list,
                       local_variable_dict,
                       episodic_buffer_dict,
                       external_textual_buffer_dict):

        chunk_num = self.get_variable_value(func_input_arg_list[0], local_variable_dict, episodic_buffer_dict)
        statement_num = self.get_variable_value(func_input_arg_list[1], local_variable_dict, episodic_buffer_dict)

        return external_textual_buffer_dict[chunk_num][statement_num]

    def judge_condition(self, current_line, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict):
        """
        This function parses the while condition line and determine whether to continue executing the program
        :param current_line:
        :param local_variable_dict:
        :return:
        """

        # TODO: this judge condition should handle both cases like a > b or a function that directly returns True/False

        function_name = StringProcessor.extract_function_name_from_string(current_line)
        if function_name is not None:
            # This handles the situation like while next_statement() ...

            func_input_arg_list = StringProcessor.extract_function_args_from_string(current_line)

            return_val = self.function_handlers[function_name](func_input_arg_list,
                                                                local_variable_dict,
                                                                episodic_buffer_dict,
                                                                external_textual_buffer_dict)

        else:
            # This handles the situation like while a > b , etc.
            tokens = StringProcessor.whitespace_tokenize_line(current_line)

            assert len(tokens) == 4, "Invalid while condition sentence!"
            var1_str = tokens[1]
            operator_str = tokens[2]
            var2_str = tokens[3]

            var1_val = self.get_variable_value(var1_str, local_variable_dict, episodic_buffer_dict)
            var2_val = self.get_variable_value(var2_str, local_variable_dict, episodic_buffer_dict)

            return_val = self.arithmetic_operators[operator_str](var1_val, var2_val)

        return return_val

    def get_while_span(self, while_span_start, program_lines):
        """
        This function gets the span of the while block
        :param while_span_counter:
        :param program_lines:
        :return: while start, while end,
        """

        while_block_start = while_span_start
        while_block_end = while_span_start + 1

        while_start_line_count = 1

        while while_start_line_count > 0 and while_block_end < len(program_lines):
            if program_lines[while_block_end].startswith("while"):
                while_start_line_count += 1

            if program_lines[while_block_end] == "end_while":
                while_start_line_count -= 1

            while_block_end += 1

        # This program lines include the "while" and "end_while" keywords
        while_block_program_lines = program_lines[while_block_start: while_block_end]

        return while_block_start, while_block_end - 1, while_block_program_lines

    def while_handler(self,
                      program_lines,
                      program_counter,
                      local_variable_dict,
                      episodic_buffer_dict,
                      external_textual_buffer_dict):
        """
        This function handles one while block
        :return:
        """

        while_block_start, while_block_end, while_block_program_lines = self.get_while_span(program_counter, program_lines)

        while_condition_judgement_line = program_lines[while_block_start]
        while self.judge_condition(
                while_condition_judgement_line, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict
        ):
            local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = \
                self.program_handler(
                    program_lines[while_block_start + 1: while_block_end],
                    local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict
                )

        program_counter = while_block_end + 1
        return program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def get_for_span(self,
                    for_span_start,
                    program_lines):

        for_block_start = for_span_start
        for_block_end = for_span_start + 1

        for_start_line_count = 1

        while for_start_line_count > 0 and for_block_end < len(program_lines):
            if program_lines[for_block_end].startswith("for"):
                for_start_line_count += 1

            if program_lines[for_block_end] == "end_for":
                for_start_line_count -= 1

            for_block_end += 1

        # This program lines include the "while" and "end_while" keywords
        for_block_program_lines = program_lines[for_block_start: for_block_end]

        return for_block_start, for_block_end - 1, for_block_program_lines

    def for_handler(self,
                    program_lines,
                    program_counter,
                    local_variable_dict,
                    episodic_buffer_dict,
                    external_textual_buffer_dict):
        """
        This function handles one for block
        :return:
        """

        for_block_start, for_block_end, for_block_program_lines = self.get_for_span(program_counter,
                                                                                    program_lines)

        for_condition_judgement_line = program_lines[for_block_start]
        loop_var_str, list_var_str = StringProcessor.extract_for_condition_vars(for_condition_judgement_line)

        list_var_vals = self.get_variable_value(list_var_str, local_variable_dict, episodic_buffer_dict)

        for loop_var_val in list_var_vals:
            local_variable_dict[loop_var_str] = loop_var_val
            local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = self.program_handler(
                program_lines[for_block_start + 1: for_block_end],
                local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict)

        program_counter = for_block_end + 1
        return program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def get_if_span(self, if_span_start, program_lines):
        """
        This function gets the span of the while block
        :param while_span_counter:
        :param program_lines:
        :return: while start, while end,
        """

        if_block_start = if_span_start
        if_block_else = if_block_start + 1
        if_block_end = if_block_start + 1

        if_start_counter_for_endif = 1
        if_start_counter_for_else = 1

        while if_start_counter_for_endif > 0 and if_block_end < len(program_lines):
            if program_lines[if_block_end].startswith("if"):
                if_start_counter_for_endif += 1

            if program_lines[if_block_end] == "end_if":
                if_start_counter_for_endif -= 1

            if_block_end += 1

            if if_start_counter_for_else > 0:
                if program_lines[if_block_else].startswith("if"):
                    if_start_counter_for_else += 1

                if program_lines[if_block_else] == "else":
                    if_start_counter_for_else -= 1

                if_block_else += 1

        if_block_program_lines = program_lines[if_block_start: if_block_end]

        return if_block_start, if_block_else - 1, if_block_end - 1, if_block_program_lines

    def if_handler(self,
                   program_lines,
                   program_counter,
                   local_variable_dict,
                   episodic_buffer_dict,
                   external_textual_buffer_dict):
        """
        This function handles one if block
        :return:
        """

        if_block_start, if_block_else, if_block_end, if_block_program_lines = self.get_if_span(program_counter, program_lines)

        # print("program lines:", program_lines)
        # print("start:", if_block_start, " else:", if_block_else, " end:", if_block_end)
        # input("----")

        if_condition_judgement_line = program_lines[if_block_start]

        if self.judge_condition(
                if_condition_judgement_line, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict
        ):
            local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = \
                self.program_handler(
                program_lines[if_block_start + 1: if_block_else],
                local_variable_dict,
                episodic_buffer_dict,
                external_textual_buffer_dict)

        else:
            local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = self.program_handler(
                program_lines[if_block_else + 1: if_block_end],
                local_variable_dict,
                episodic_buffer_dict,
                external_textual_buffer_dict)

        program_counter = if_block_end + 1
        return program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def arith_sum_handler(self,
                          func_input_arg_list,
                          local_variable_dict,
                          episodic_buffer_dict,
                          external_textual_buffer_dict
                          ):

        var_vals = []
        for var_str in func_input_arg_list:

            var_val = self.get_variable_value(var_str, local_variable_dict, episodic_buffer_dict)

            if isinstance(var_val, str):
                if var_val.replace("-", "").isdigit():
                    var_val = int(var_val)

            if isinstance(var_val, int):
                var_vals.append(var_val)

        return sum(var_vals)

    def add_to_episodic_handler(self,
                                program_lines,
                                program_counter,
                                local_variable_dict,
                                episodic_buffer_dict,
                                external_textual_buffer_dict
                                ):

        current_line = program_lines[program_counter]
        func_input_arg_list = StringProcessor.extract_function_args_from_string(current_line)

        epi_location_counter = len(episodic_buffer_dict)

        for var_str in func_input_arg_list:
            var_val = self.get_variable_value(var_str, local_variable_dict, episodic_buffer_dict)

            episodic_buffer_dict["episodic_buffer_" + str(epi_location_counter)] = var_val
            epi_location_counter += 1

        program_counter += 1
        return program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def clear_mem_handler(self,
                          program_lines,
                          program_counter,
                          local_variable_dict,
                          episodic_buffer_dict,
                          external_textual_buffer_dict
                          ):

        input_text = "clear_mem: " + \
                     " ".join([k + ": " + str(v) for k, v in episodic_buffer_dict.items()])

        cleared_mem_str = self.neural_module.inference(input_text)
        episodic_buffer_str_list = StringProcessor.get_episodic_buffer_str_list(cleared_mem_str)

        cleared_mem_list = [
            self.get_variable_value(s, local_variable_dict, episodic_buffer_dict) for s in episodic_buffer_str_list
        ]

        episodic_buffer_dict = {"episodic_buffer_" + str(idx): epi_str
                                for idx, epi_str in enumerate(cleared_mem_list)}

        return program_counter + 1, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def append_to_list_handler(self,
                               func_input_arg_list,
                               local_variable_dict,
                               episodic_buffer_dict,
                               external_textual_buffer_dict):

        subject_list = self.get_variable_value(func_input_arg_list[0], local_variable_dict, episodic_buffer_dict)

        if len(func_input_arg_list) > 1:
            for var_str in func_input_arg_list[1:]:
                subject_list.append(self.get_variable_value(var_str, local_variable_dict, episodic_buffer_dict))

        return subject_list

    def list_chunk_nums_handler(self,
                                func_input_arg_list,
                                local_variable_dict,
                                episodic_buffer_dict,
                                external_textual_buffer_dict
                                ):

        assert len(func_input_arg_list) == 2
        func_arg_val_list = [
            self.get_variable_value(var_str, local_variable_dict, episodic_buffer_dict)
            for var_str in func_input_arg_list
        ]

        extract_pattern = r'chunk_(\d+)'
        start_num = int(re.findall(extract_pattern, func_arg_val_list[0])[0])
        end_num = int(re.findall(extract_pattern, func_arg_val_list[1])[0])

        chunk_list = []
        for chunk_num in range(start_num, end_num + 1):
            chunk_list.append("chunk_" + str(chunk_num))

        return chunk_list

    def generate_program(self, episodic_buffer_dict):

        input_text = "generate_program: " + \
            " ".join([k + ": " + str(v) for k, v in episodic_buffer_dict.items()])

        program_text = self.neural_module.inference(input_text)

        program_lines = program_text.split(";")

        program_lines = [line for line in program_lines if (line != "" and not line.isspace())]

        program_lines = [StringProcessor.remove_string_leading_spaces(line) for line in program_lines]

        return program_lines

    def new_mem_handler(self,
                        program_lines_parent_level,
                        program_counter_parent_level,
                        local_variable_dict_parent_level,
                        episodic_buffer_dict_parent_level,
                        external_textual_buffer_dict):

        """
        This function is probably the most confusing function in the interpreter: it basically handles a recursive
        call. It contains the lines, local variable buffer and episodic buffer for both the parent and the child level.

        The parent level local variable buffer and episodic buffer are copied. After the execution, the parent level
        local variable buffer should not be modified, and the parent level episodic buffer should be updated.

        The child level local variable buffer should be initiated empty, and the child level episodic buffer should
        only contain the content that's meant to be copied from the parent level process.
        :param program_lines_parent_level:
        :param program_counter_parent_level:
        :param local_variable_dict_parent_level:
        :param episodic_buffer_dict_parent_level:
        :param external_textual_buffer_dict:
        :return:
        """

        # Initiate a new episodic memory buffer of the child level.
        if len(program_lines_parent_level) > 0:
            current_line = program_lines_parent_level[program_counter_parent_level]
            func_input_arg_list = StringProcessor.extract_function_args_from_string(current_line)
        else:
            func_input_arg_list = []

        episodic_buffer_dict_child_level = {}
        epi_location_counter = 0
        for var_str in func_input_arg_list:
            var_val = self.get_variable_value(
                var_str, local_variable_dict_parent_level, episodic_buffer_dict_parent_level
            )

            episodic_buffer_dict_child_level["episodic_buffer_" + str(epi_location_counter)] = var_val
            epi_location_counter += 1

        # There are different design philosophies to handle the local variable dict: allow it to be accessed in
        # different loops of the same level, or do not allow it. Currently we allow it to be accessed in different
        # loops.
        local_variable_dict_child_level = {}

        self.indent_level += 1
        for i in range(10):

            if self.debug_flag:
                print("=" * 40)
                print("\t" * self.indent_level,
                      "indent level", self.indent_level,
                      " loop idx:", i)
                UtilsEVRDebug.debug_print_episodic_buffer(
                    episodic_buffer_dict_child_level, indent_level=self.indent_level,
                    caption="episodic buffer before executing the program"
                )

            # First call the program generator to generate the programs.
            program_lines_child_level = self.generate_program(episodic_buffer_dict_child_level)
            program_lines_keywords = [self.get_line_keyword(line) for line in program_lines_child_level]

            if self.debug_flag:
                UtilsEVRDebug.debug_print_program_lines(program_lines_child_level,
                                                        self.indent_level)

            if "return" in program_lines_keywords:
                return_line_idx = program_lines_keywords.index("return")

                local_variable_dict_child_level, episodic_buffer_dict_child_level, external_textual_buffer_dict = \
                    self.program_handler(
                        program_lines=program_lines_child_level[:return_line_idx],
                        local_variable_dict=local_variable_dict_child_level,
                        episodic_buffer_dict=episodic_buffer_dict_child_level,
                        external_textual_buffer_dict=external_textual_buffer_dict
                    )

                episodic_buffer_dict_parent_level = self.return_handler(
                    program_lines_child_level[return_line_idx],
                    local_variable_dict_child_level,
                    episodic_buffer_dict_child_level,
                    episodic_buffer_dict_parent_level
                )

                if self.debug_flag:
                    UtilsEVRDebug.debug_print_episodic_buffer(
                        episodic_buffer_dict_child_level, indent_level=self.indent_level,
                        caption="episodic buffer after executing the program"
                    )
                    UtilsEVRDebug.debug_print_local_variable_dict(
                        local_variable_dict_child_level, indent_level=self.indent_level)

                    if self.print_chunk:
                        UtilsEVRDebug.debug_print_external_textual_buffer(
                            external_textual_buffer_dict, indent_level=self.indent_level)
                    input("=" * 40)

                break   # If return in the lines, terminate loop immediately

            else:
                local_variable_dict_child_level, episodic_buffer_dict_child_level, external_textual_buffer_dict = \
                    self.program_handler(
                        program_lines=program_lines_child_level,
                        local_variable_dict=local_variable_dict_child_level,
                        episodic_buffer_dict=episodic_buffer_dict_child_level,
                        external_textual_buffer_dict=external_textual_buffer_dict
                    )

                if self.debug_flag:
                    UtilsEVRDebug.debug_print_episodic_buffer(
                        episodic_buffer_dict_child_level, indent_level=self.indent_level,
                        caption="episodic buffer after executing the program")
                    UtilsEVRDebug.debug_print_local_variable_dict(
                        local_variable_dict_child_level, indent_level=self.indent_level)

                    if self.print_chunk:
                        UtilsEVRDebug.debug_print_external_textual_buffer(
                            external_textual_buffer_dict, indent_level=self.indent_level)

                    input("=" * 40)

        program_counter_parent_level += 1
        self.indent_level -= 1
        return program_counter_parent_level, local_variable_dict_parent_level, \
               episodic_buffer_dict_parent_level, external_textual_buffer_dict

    def return_handler(self,
                       current_line_child_level,
                        local_variable_dict_child_level,
                        episodic_buffer_dict_child_level,
                       episodic_buffer_dict_parent_level):

        func_input_arg_list = StringProcessor.extract_function_args_from_string(current_line_child_level)

        epi_location_counter = len(episodic_buffer_dict_parent_level)

        # Add the variable values to the parent level episodic buffer dict
        for var_str in func_input_arg_list:
            var_val = self.get_variable_value(
                var_str,
                local_variable_dict_child_level,
                episodic_buffer_dict_child_level
            )

            episodic_buffer_dict_parent_level["episodic_buffer_" + str(epi_location_counter)] = var_val
            epi_location_counter += 1

        # It is OK that this function does not return program counter and local variable dict, because this function
        # will never be handled by a program handler.
        return episodic_buffer_dict_parent_level

    def del_handler(self,
                    program_lines,
                    program_counter,
                    local_variable_dict,
                    episodic_buffer_dict,
                    external_textual_buffer_dict
                    ):

        current_line = program_lines[program_counter]
        func_input_arg_list = StringProcessor.extract_function_args_from_string(current_line)
        func_arg_val_list = [
            self.get_variable_value(var_str, local_variable_dict, episodic_buffer_dict)
            for var_str in func_input_arg_list
        ]

        for arg_name in func_arg_val_list:
            if arg_name in local_variable_dict:
                del local_variable_dict[arg_name]

            if arg_name in episodic_buffer_dict:
                del episodic_buffer_dict[arg_name]

            if arg_name in external_textual_buffer_dict:
                del external_textual_buffer_dict[arg_name]

        return program_counter + 1, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def update_chunk_handler(self,
                             program_lines,
                             program_counter,
                             local_variable_dict,
                             episodic_buffer_dict,
                             external_textual_buffer_dict
                             ):

        current_line = program_lines[program_counter]
        func_input_arg_list = StringProcessor.extract_function_args_from_string(current_line)

        func_arg_val_list = [
            self.get_variable_value(var_str, local_variable_dict, episodic_buffer_dict)
            for var_str in func_input_arg_list
        ]

        chunk_name = func_arg_val_list[0]
        chunk_val = func_arg_val_list[1]

        if isinstance(chunk_val, int) or isinstance(chunk_val, bool) or isinstance(chunk_val, str):

            external_textual_buffer_dict[chunk_name] = chunk_val

        if isinstance(chunk_val, list):
            external_textual_buffer_dict[chunk_name] = {
                "statement_" + str(ele_idx): chunk_val_one_ele for ele_idx, chunk_val_one_ele in enumerate(chunk_val)
            }

        return program_counter + 1, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def clean_chunks_handler(self,
                             program_lines,
                             program_counter,
                             local_variable_dict,
                             episodic_buffer_dict,
                             external_textual_buffer_dict
                             ):

        new_external_buffer_list = []

        for external_buffer_item in external_textual_buffer_dict.values():
            if len(external_buffer_item) > 3:
                n_new_chunk = math.ceil(len(external_buffer_item) / 3)
                for i in range(n_new_chunk):  # loop over the new chunks
                    new_chunk = {}
                    for j in range(3 * i, min(3 * (i + 1), len(external_buffer_item))):
                        new_chunk["statement_" + str(j % 3)] = external_buffer_item["statement_" + str(j)]
                    new_external_buffer_list.append(new_chunk)
            else:
                new_external_buffer_list.append(external_buffer_item)

        external_textual_buffer_dict = {
            "chunk_" + str(idx): new_chunk
            for idx, new_chunk in enumerate(new_external_buffer_list)
        }

        return program_counter + 1, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict

    def get_line_keyword(self, program_line):
        """
        This function parses each line of code, and decide which function to be further used to execute the program.
        It should determine whether the function is While, For, If, or other functions.
        :return:
        """

        tokens = StringProcessor.whitespace_tokenize_line(program_line)

        # There should be only three types of commands: assignment, control flow and functions.
        if len(tokens) >= 2 and tokens[1] == "=":
            keyword_name = "assign"
        elif tokens[0] in ["if", "for", "while"]:
            keyword_name = tokens[0]
        elif tokens[0] == "pass":
            keyword_name = "pass"
        else:
            keyword_name = StringProcessor.extract_function_name_from_string(program_line)

        return keyword_name

    def program_handler(self, program_lines, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict):
        """
        Loop over all the lines of the program and call the suitable function handler to interpret the code.
        :param program_lines:
        :param local_variable_dict:
        :param episodic_buffer_dict:
        :param external_textual_buffer_dict:
        :return:
        """

        program_counter = 0
        while program_counter <= len(program_lines) - 1:
            keyword_name = self.get_line_keyword(program_lines[program_counter])

            program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = \
                self.keyword_handlers[keyword_name](program_lines,
                                                    program_counter,
                                                    local_variable_dict,
                                                    episodic_buffer_dict,
                                                    external_textual_buffer_dict)

            # except:
            #     print("*" * 40)
            #     print("Error!")
            #     print("line " + str(program_counter - 1) + " line:" + program_lines[program_counter - 1])
            #     print("*" * 40)
            #
            #     assert False, "Program Stopped!"
            # print(keyword_name, program_counter, local_variable_dict)
            # input("-----")

        return local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict


