from preliminary_experiments.experiments_evr.evr_class.utils_evr_string_processing import StringProcessor


class CheckStringProcessor:

    @classmethod
    def check_whitespace_tokenize(cls):

        strs = [
            "#1 = ABC(d)",
            "#2 = #3",
            "#3 = ABCD(d, e, f, g)",
            " #4  = 5"
        ]

        print("=" * 40)
        print("check white space tokenize")
        print("=" * 40)
        for s in strs:
            print("-" * 40)
            print(s)
            print(StringProcessor.whitespace_tokenize_line(s))

    @classmethod
    def check_removing_leading_spaces(cls):

        strs = [
            " #1 = ABC(d)",
            "#2 = #3 ",
            "   #3 = ABCD(d, e, f, g)     ",
            "      #4  = 5 ",
            "   ",
            ""
        ]

        print("=" * 40)
        print("check leading space removing")
        print("=" * 40)
        for s in strs:
            print("-" * 40)
            print(s)
            print(StringProcessor.remove_string_leading_spaces(s))

    @classmethod
    def check_extract_local_variable_name_from_string(cls):
        strs = [
            " #1 ",
            "#2",
            "#33",
            "#567_b",
            "8"
        ]

        print("=" * 40)
        print("check extracting local variable name")
        print("=" * 40)
        for s in strs:
            print("-" * 40)
            print(s)
            print(StringProcessor.extract_local_variable_name_from_string(s))

    @classmethod
    def check_extract_function_name_from_string(cls):
        strs = [
            " #1 = ABC(d)",
            "   #3 = ABCD(d, e, f, g)     ",
            " #2 = abc_123(#1, e, #33, episodic_buffer)",
            " 5 = abc_123(#1, e, #33, episodic_buffer)",
            "ab=abc_123(#1, e, #33, episodic_buffer)",
            "#5 = 6"
        ]

        print("=" * 40)
        print("check extracting function name")
        print("=" * 40)
        for s in strs:
            print("-" * 40)
            print(s)
            print(StringProcessor.extract_function_name_from_string(s))

    @classmethod
    def check_extract_function_arg_list_from_string(cls):
        strs = [
            " #1 = ABC(d)",
            "#1=ABC()",
            "   #3 = ABCD(d, e, f, g)     ",
            " #2 = abc_123(#1, e, #33, episodic_buffer)",
            " 5 = abc_123(#1, e, #33, episodic_buffer)",
            "ab=abc_123(#1, e, #33, episodic_buffer)",
            "#5 = 6",
        ]

        print("=" * 40)
        print("check extracting function name")
        print("=" * 40)
        for s in strs:
            args = StringProcessor.extract_function_args_from_string(s)

            print("-" * 40)
            print(s)
            print(args)
            assert "" not in args

    @classmethod
    def check_list_string_processing(cls):

        cases = [
            "[]",
            "[,]",
            "[a]",
            "[0]",
            "['x']",
            "['x', ]",
            "[ 'x'  ,  ]",
            "[  'x', True ,'True',  0 , episodic_buffer_2,  5]"
        ]

        targets = [
            [],
            [],
            ['a'],
            ['0'],
            ["'x'"],
            ["'x'"],
            ["'x'"],
            ["'x'", 'True', "'True'", '0', 'episodic_buffer_2', '5']
        ]

        for i, case in enumerate(cases):
            assert targets[i] == StringProcessor.process_list(case)

    @classmethod
    def check_for_loop_extract(cls):

        cases = [
            "for #1 in x",
            " for  #1  in   x    ",
            "for #11 in [1,2]",
            "  for   #12    in    []"
        ]

        targets = [
            ("#1", "x"),
            ("#1", "x"),
            ("#11", "[1,2]"),
            ("#12", "[]")
        ]

        for i, case in enumerate(cases):
            loop_var, list_var = StringProcessor.extract_for_condition_vars(case)

            print(loop_var, list_var)

            assert loop_var == targets[i][0], list_var == targets[i][1]


if __name__ == "__main__":

    # CheckInterpreterStringProcessor.check_whitespace_tokenize()
    # CheckInterpreterStringProcessor.check_removing_leading_spaces()
    # CheckInterpreterStringProcessor.check_extract_local_variable_name_from_string()
    # CheckInterpreterStringProcessor.check_extract_function_name_from_string()
    # CheckInterpreterStringProcessor.check_extract_function_arg_list_from_string()
    # CheckInterpreterStringProcessor.check_list_string_processing()
    CheckStringProcessor.check_for_loop_extract()
