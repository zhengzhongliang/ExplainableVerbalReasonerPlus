import unittest
from preliminary_experiments.data_generation.data_1_cartesian_evr import GenerateEVRCartesianData


class TestGenerateCartesianData(unittest.TestCase):

    test_instance = {
        "id": "405ce9022d33bca3664e2eca16baaae6",
        "depth": 2,
        "context_string": "Each of Justin Wright and Richard Adams had 5 apples and 13 owls.",
        "question_string": "List the items and the number of each item each person had.",
        "answer": ("Justin Wright had 5 apples, Justin Wright had 13 owls, "
                   "Richard Adams had 5 apples, Richard Adams had 13 owls."),
        "target_list": [
            ["Justin Wright", 5, ["apple", "apples"]],
            ["Justin Wright", 13, ["owl", "owls"]],
            ["Richard Adams", 5, ["apple", "apples"]],
            ["Richard Adams", 13, ["owl", "owls"]]
        ],
        "target_nl_list": [
            "Justin Wright had 5 apples",
            "Justin Wright had 13 owls",
            "Richard Adams had 5 apples",
            "Richard Adams had 13 owls"
        ],
    }

    def test_generate_program(self, debug_flag=False, pattern="generate_program", pattern_num=12):
        evr_instances = GenerateEVRCartesianData.generate_evr_data_one_instance(self.test_instance)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Pattern 1
        assert (
            (
                ("generate_program: "
                 "episodic_buffer_0: This is a cartesian task. "
                 "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                 "episodic_buffer_2: List the items that each person had."),
                ("#0 = 'Who are the persons?'; "
                 "new_mem(episodic_buffer_1, #0);")
            ) in in_out_pairs
        )

        # Pattern 2
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_1: Who are the persons?"),
                    ("#0 = get_chunk('chunk_0'); "
                     "#1 = qa(#0, episodic_buffer_1); "
                     "add_to_episodic(#1);")
                ) in in_out_pairs
        )

        # Pattern 3
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_1: Who are the persons? "
                     "episodic_buffer_2: The persons are ['Justin Wright', 'Richard Adams']."),
                    "return(episodic_buffer_2);"
                ) in in_out_pairs
        )

        # Pattern 4
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The persons are ['Justin Wright', 'Richard Adams']."),
                    ("#0 = 'What are the items?'; "
                     "new_mem(episodic_buffer_1, #0);")
                ) in in_out_pairs
        )

        # Pattern 5
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_1: What are the items?"),
                    ("#0 = get_chunk('chunk_0'); "
                     "#1 = qa(#0, episodic_buffer_1); "
                     "add_to_episodic(#1);")
                ) in in_out_pairs
        )

        # Pattern 6
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_1: What are the items? "
                     "episodic_buffer_2: The items are ['5 apples', '13 owls']."),
                    "return(episodic_buffer_2);"
                ) in in_out_pairs
        )

        # Pattern 7
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The persons are ['Justin Wright', 'Richard Adams']. "
                     "episodic_buffer_4: The items are ['5 apples', '13 owls']."),
                    ("#0 = ['Justin Wright', 'Richard Adams']; "
                     "add_to_episodic('#0 stores the list of persons.');")
                ) in in_out_pairs
        )

        # Pattern 8
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The persons are ['Justin Wright', 'Richard Adams']. "
                     "episodic_buffer_4: The items are ['5 apples', '13 owls']. "
                     "episodic_buffer_5: #0 stores the list of persons."),
                    "clear_mem();"
                ) in in_out_pairs
        )

        # Pattern 9
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The items are ['5 apples', '13 owls']. "
                     "episodic_buffer_4: #0 stores the list of persons."),
                    ("#1 = ['5 apples', '13 owls']; "
                     "add_to_episodic('#1 stores the list of items.');")
                ) in in_out_pairs
        )

        # Pattern 10
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The items are ['5 apples', '13 owls']. "
                     "episodic_buffer_4: #0 stores the list of persons. "
                     "episodic_buffer_5: #1 stores the list of items."),
                    "clear_mem();"
                ) in in_out_pairs
        )

        # Pattern 11
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: #0 stores the list of persons. "
                     "episodic_buffer_4: #1 stores the list of items."),
                    ("#2 = []; "
                     "for #3 in #0; "
                     "for #4 in #1; "
                     "#5 = 'How many items did this person have?'; "
                     "#6 = rewrite(#5, #3, #4); "
                     "#2 = append_to_list(#2, #6); "
                     "end_for; "
                     "end_for; "
                     #"#7 = rewrite('Change the list to a natural language sentence', #2); "
                     "return(#2);")
                ) in in_out_pairs
        )

        # Pattern 12
        # assert (
        #         (
        #             ("generate_program: "
        #              "episodic_buffer_0: This is a cartesian task. "
        #              "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
        #              "episodic_buffer_2: List the items that each person had. "
        #              "episodic_buffer_3: #0 stores the list of persons. "
        #              "episodic_buffer_4: #1 stores the list of items. "
        #              "episodic_buffer_5: Justin Wright had 5 apples, Justin Wright had 13 owls, "
        #                 "Richard Adams had 5 apples, Richard Adams had 13 owls."),
        #             "return(episodic_buffer_5);"
        #         ) in in_out_pairs
        # )

    def test_rewrite(self, debug_flag=False, pattern="rewrite", pattern_num=2):
        evr_instances = GenerateEVRCartesianData.generate_evr_data_one_instance(self.test_instance)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Pattern 1
        assert (
                (
                    ("rewrite: "
                     "How many items did this person have? Justin Wright. 5 apples."),
                    "Justin Wright had 5 apples."
                ) in in_out_pairs
        )

        assert (
                (
                    ("rewrite: "
                     "How many items did this person have? Richard Adams. 13 owls."),
                    "Richard Adams had 13 owls."
                ) in in_out_pairs
        )

        # Pattern 2
        # assert (
        #         (
        #             ("rewrite: "
        #              "Change the list to a natural language sentence. "
        #              "Justin Wright had 5 apples. Justin Wright had 13 owls. "
        #              "Richard Adams had 5 apples. Richard Adams had 13 owls."),
        #             ("Justin Wright had 5 apples, Justin Wright had 13 owls, "
        #              "Richard Adams had 5 apples, Richard Adams had 13 owls.")
        #         ) in in_out_pairs
        # )

    def test_qa(self, debug_flag=False, pattern="qa", pattern_num=1):
        evr_instances = GenerateEVRCartesianData.generate_evr_data_one_instance(self.test_instance)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Pattern 1
        assert (
                (
                    ("qa: "
                     "statement_0: Each of Justin Wright and Richard Adams had 5 apples and 13 owls. "
                     "Who are the persons?"),
                    "The persons are ['Justin Wright', 'Richard Adams']."
                ) in in_out_pairs
        )

        # Pattern 2
        assert (
                (
                    ("qa: "
                     "statement_0: Each of Justin Wright and Richard Adams had 5 apples and 13 owls. "
                     "What are the items?"),
                    "The items are ['5 apples', '13 owls']."
                ) in in_out_pairs
        )

    def test_clear_mem(self, debug_flag=True, pattern="clear_mem", pattern_num=1):
        evr_instances = GenerateEVRCartesianData.generate_evr_data_one_instance(self.test_instance)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Pattern 1
        assert (
                (
                    ("clear_mem: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The persons are ['Justin Wright', 'Richard Adams']. "
                     "episodic_buffer_4: The items are ['5 apples', '13 owls']. "
                     "episodic_buffer_5: #0 stores the list of persons."),
                    ("episodic_buffer_0: 'This is a cartesian task.' "
                     "episodic_buffer_1: 'Chunk 0 can be used to infer the number of items each person had.' "
                     "episodic_buffer_2: 'List the items that each person had.' "
                     "episodic_buffer_3: 'The items are ['5 apples', '13 owls'].' "
                     "episodic_buffer_4: '#0 stores the list of persons.'")
                ) in in_out_pairs
        )

        # Pattern 2
        assert (
                (
                    ("clear_mem: "
                     "episodic_buffer_0: This is a cartesian task. "
                     "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                     "episodic_buffer_2: List the items that each person had. "
                     "episodic_buffer_3: The items are ['5 apples', '13 owls']. "
                     "episodic_buffer_4: #0 stores the list of persons. "
                     "episodic_buffer_5: #1 stores the list of items."),
                    ("episodic_buffer_0: 'This is a cartesian task.' "
                     "episodic_buffer_1: 'Chunk 0 can be used to infer the number of items each person had.' "
                     "episodic_buffer_2: 'List the items that each person had.' "
                     "episodic_buffer_3: '#0 stores the list of persons.' "
                     "episodic_buffer_4: '#1 stores the list of items.'")
                ) in in_out_pairs
        )


if __name__ == "__main__":
    unittest.main()
