from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent


class CheckChunkOperation:

    @classmethod
    def check_list_chunk_nums(cls):

        evr_agent = EVRAgent(neural_module=None)

        chunk_num_list = evr_agent.list_chunk_nums_handler(
            ["'chunk_0'", "'chunk_2'"],
            local_variable_dict={},
            episodic_buffer_dict={},
            external_textual_buffer_dict={}
        )

        print(chunk_num_list)

    @classmethod
    def check_update_chunk(cls):
        evr_agent = EVRAgent(neural_module=None)

        external_textual_buffer_dict = {"chunk_0": {"statement_0": 1}}

        print("=" * 40)
        print("external buffer old:")
        print(external_textual_buffer_dict)
        print("=" * 40)

        program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = \
            evr_agent.update_chunk_handler(
                program_lines=["update_chunk('chunk_0', #0)"],
                program_counter=0,
                local_variable_dict={"#0": ['a', 'b', 'c']},
                episodic_buffer_dict={},
                external_textual_buffer_dict=external_textual_buffer_dict
            )

        print("external buffer new:")
        print(external_textual_buffer_dict)

    @classmethod
    def check_clean_chunks_handler(cls):
        evr_agent = EVRAgent(neural_module=None)

        external_textual_buffer_dict = {
            "chunk_0": {
                "statement_0": 1,
                "statement_1": 2,
                "statement_2": 3,
                "statement_3": 4,
                "statement_4": 5,
            },
            "chunk_1": {
                "statement_0": 6,
                "statement_1": 7,
                "statement_2": 8,
            },
            "chunk_2": {
                "statement_0": 9,
            },
        }

        print("=" * 40)
        print("external buffer old:")
        print(external_textual_buffer_dict)
        print("=" * 40)

        program_counter, local_variable_dict, episodic_buffer_dict, external_textual_buffer_dict = \
            evr_agent.clean_chunks_handler(
                program_lines=["clean_chunks()"],
                program_counter=0,
                local_variable_dict={"#0": ['a', 'b', 'c']},
                episodic_buffer_dict={},
                external_textual_buffer_dict=external_textual_buffer_dict
            )

        print("external buffer new:")
        print(external_textual_buffer_dict)


if __name__ == "__main__":

    CheckChunkOperation.check_list_chunk_nums()

    CheckChunkOperation.check_update_chunk()

    CheckChunkOperation.check_clean_chunks_handler()
