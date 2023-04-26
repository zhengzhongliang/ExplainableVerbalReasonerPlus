

class UtilsEVRDebug:

    @staticmethod
    def debug_print_episodic_buffer(episodic_buffer_dict, indent_level, caption=None):

        print("\t" * indent_level, "-" * 20)
        if caption:
            print("\t" * indent_level, caption)

        for mem_key, mem_content in episodic_buffer_dict.items():
            print("\t" * indent_level, mem_key, ":", mem_content)

        print("\t" * indent_level, "-" * 20)

    @staticmethod
    def debug_print_local_variable_dict(local_variable_dict, indent_level):
        print("\t" * indent_level, "-" * 20)

        for var_name, var_value in local_variable_dict.items():
            print("\t" * indent_level, var_name, ":", var_value)

        print("\t" * indent_level, "-" * 20)

    @staticmethod
    def debug_print_program_lines(program_lines, indent_level):

        print("\t" * indent_level, "-" * 20)

        for program_line in program_lines:
            print("\t" * indent_level, program_line)

        print("\t" * indent_level, "-" * 20)

    @staticmethod
    def debug_print_external_textual_buffer(external_textual_buffer, indent_level):

        print("\t" * indent_level, "-" * 20)
        print("\t" * indent_level, external_textual_buffer)

        print("\t" * indent_level, "-" * 20)