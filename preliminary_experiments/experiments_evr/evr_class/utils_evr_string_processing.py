import re


class StringProcessor:

    @classmethod
    def whitespace_tokenize_line(cls, program_line):

        program_line = cls.remove_string_leading_spaces(program_line)

        tokens = program_line.split(" ")
        tokens = [t for t in tokens if t != ""]

        return tokens

    @classmethod
    def remove_string_leading_spaces(cls, input_string):

        start_idx = 0
        end_idx = len(input_string)

        while start_idx < len(input_string) and input_string[start_idx] == " ":
            start_idx += 1

        while end_idx > 0 and input_string[end_idx - 1] == " ":
            end_idx -= 1

        if start_idx > end_idx or input_string[start_idx: end_idx] == "":
            return None
        else:
            return input_string[start_idx: end_idx]

    @classmethod
    def extract_local_variable_name_from_string(cls, input_string):

        local_var_name = re.findall(r'(#[0-9]+)', input_string)

        if len(local_var_name) > 0:
            return local_var_name[0]

        else:
            return None

    @classmethod
    def extract_function_name_from_string(cls, input_string):

        match_pattern = "([a-zA-Z0-9_]+)\(.*\)"

        function_names = re.findall(match_pattern, input_string)

        if len(function_names) > 0:
            return function_names[0]
        else:
            return None

    @classmethod
    def extract_function_args_from_string(cls, input_string):

        match_pattern = "[a-zA-Z0-9_]+\((.*)\)"

        vars_string = re.findall(match_pattern, input_string)

        if len(vars_string):

            var_list = vars_string[0].split(",")

            return [StringProcessor.remove_string_leading_spaces(v) for v in var_list if v != "" and not v.isspace()]

        else:
            return []

    @classmethod
    def form_textual_input_from_chunk_and_statements(cls, chunk_or_statement):

        if isinstance(chunk_or_statement, str):
            return str(chunk_or_statement)

        elif isinstance(chunk_or_statement, int):
            return str(chunk_or_statement)

        elif isinstance(chunk_or_statement, list):
            return " ".join(chunk_or_statement)

        elif isinstance(chunk_or_statement, dict):
            return " ".join([str(k) + ": " + str(v) for k, v in chunk_or_statement.items()])

        else:
            return ""

    @classmethod
    def process_list(cls, list_str):

        list_str_no_braces = list_str[1:-1]
        var_strs = list_str_no_braces.split(",")
        var_strs = [var_str for var_str in var_strs if var_str != "" and not var_str.isspace()]

        var_strs = [cls.remove_string_leading_spaces(var_str) for var_str in var_strs]

        return var_strs

    @classmethod
    def extract_for_condition_vars(cls, for_condition_line):

        loop_var_pattern = r'for\s+(#[0-9]+)\s+in'
        list_var_pattern = r'for\s+#[0-9]+\s+in\s+(.+)'

        loop_var_str = re.findall(loop_var_pattern, for_condition_line)[0]
        list_var_str = re.findall(list_var_pattern, for_condition_line)[0]

        return loop_var_str, list_var_str

    @classmethod
    def get_episodic_buffer_str_list(cls, episodic_buffer_str):

        ep_strs = re.split(r'episodic_buffer_\d+: ', episodic_buffer_str)

        ep_strs = [s for s in ep_strs if not s.isspace() and s != ""]

        return ep_strs
