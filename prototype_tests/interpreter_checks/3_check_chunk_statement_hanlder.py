from preliminary_experiments.experiments_evr.evr_class.InterpreterClass import ProgramInterpreter


class CheckChunkStatementHandler:

    @classmethod
    def check_next_chunk(cls):

        prog_int = ProgramInterpreter(neural_module=None)

        ext_mems = [
            {},
            {1: [1, 2], 2: [3, 4]},
            {"1": [1, 2], "2": [3, 4]}
        ]

        print("=" * 40)
        print("checking next chunk func")
        print("=" * 40)
        for ext_mem in ext_mems:

            print("-" * 40)
            print("chunk:", ext_mem)
            print("-" * 40)
            local_variable_dict = {}

            for i in range(5):
                next_chunk_exist = prog_int.check_next_chunk_handler(func_input_arg_list=[],
                                                    local_variable_dict=local_variable_dict,
                                                    episodic_buffer_dict={},
                                                    external_textual_buffer_dict=ext_mem)

                next_chunk_num = prog_int.get_next_chunk_num_handler(func_input_arg_list=[],
                                                    local_variable_dict=local_variable_dict,
                                                    episodic_buffer_dict={},
                                                    external_textual_buffer_dict=ext_mem)

                print("\tnext chunk exists?", next_chunk_exist)
                print("\tnext chunk num?", next_chunk_num)
                print("\tlocal var:", local_variable_dict)

                if next_chunk_num is not None:
                    if isinstance(next_chunk_num, str):
                        local_variable_dict["#1"] = next_chunk_num[1: -1]
                    else:
                        local_variable_dict["#1"] = next_chunk_num

                    retrieved_chunk = prog_int.get_chunk_handler(["#1"], local_variable_dict, {}, ext_mem)

                    print("\tretrieved chunk:", retrieved_chunk)
                print("-" * 40)

    @classmethod
    def check_next_statement(cls, chunk_key):

        prog_int = ProgramInterpreter(neural_module=None)

        ext_mems = [
            {},
            {1: {}, 2: {21: [21], 22: [22]}},
            {"1": {}, "2": {"21": [21], "22": [22]}},
            {"1": {}, "2": {21: [21], "22": [22]}}
        ]

        for ext_mem in ext_mems:

            print("=" * 40)
            print("chunk:", ext_mem)
            print("chunk key:", chunk_key)
            print("=" * 40)
            local_variable_dict = {}

            for i in range(5):
                next_statement_exist = prog_int.check_next_statement_handler(func_input_arg_list=[chunk_key],
                                                                     local_variable_dict=local_variable_dict,
                                                                     episodic_buffer_dict={},
                                                                     external_textual_buffer_dict=ext_mem)

                next_statement_num = prog_int.get_next_statement_num_handler(func_input_arg_list=[chunk_key],
                                                                     local_variable_dict=local_variable_dict,
                                                                     episodic_buffer_dict={},
                                                                     external_textual_buffer_dict=ext_mem)

                print("\tnext statement exists?", next_statement_exist)
                print("\tnext statement num?", next_statement_num)
                print("\tlocal var:", local_variable_dict)

                if next_statement_num is not None:
                    if isinstance(next_statement_num, str):
                        local_variable_dict["#1"] = next_statement_num[1: -1]
                    else:
                        local_variable_dict["#1"] = next_statement_num

                    retrieved_chunk = prog_int.get_statement_handler([chunk_key, "#1"], local_variable_dict, {}, ext_mem)

                    print("\tretrieved statement:", retrieved_chunk)
                print("-" * 40)


if __name__ == "__main__":

    #CheckChunkStatementHandler.check_next_chunk()

    for chunk_key in ["1", "2", "'1'", "'2'"]:
        CheckChunkStatementHandler.check_next_statement(chunk_key)
