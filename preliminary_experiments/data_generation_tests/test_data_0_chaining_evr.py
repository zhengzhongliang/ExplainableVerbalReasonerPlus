import unittest
from preliminary_experiments.data_generation.data_0_chaining_evr import GenerateEVRChainingData


class TestGenerateTreeSearchData(unittest.TestCase):

    test_instances = [
        {
            "id": "dd04ad15c157758147de6b343db4fbfb",
            "chains": [
                {
                    "formal_reps": [
                        ["Carl Patel", 19, ["toy car", "toy cars"]],
                    ],
                    "quantity_ops": [19, -1, 1],
                    "context_list": [
                        "Carl Patel had 19 toy cars in the beginning.",
                    ],
                    "answer": 19
                },
                {
                    "formal_reps": [
                        ["Carl Patel", 2, ["toy bear", "toy bears"]],
                    ],
                    "quantity_ops": [2, 1, 2],
                    "context_list": [
                        "Carl Patel had 2 toy bears in the beginning.",
                    ],
                    "answer": 2
                },
                {
                    "formal_reps": [
                        ["Paul Cook", 16, ["toy bear", "toy bears"]],
                    ],
                    "quantity_ops": [16, 0, 2],
                    "context_list": [
                        "Paul Cook had 16 toy bears in the beginning.",
                    ],
                    "answer": 16
                }
            ],
            "selected_chain_idx": 2,
            "context_string": ("Carl Patel had 19 toy cars in the beginning. "
                               "Carl Patel had 2 toy bears in the beginning. "
                               "Paul Cook had 16 toy bears in the beginning."),
            "question_string": "How many toy bears did Paul Cook have in the end?",
            "answer": 16,
            "depth": 0
        },
        {
            "id": "dd04ad15c157758147de6b343db4fbfb",
            "chains": [
                {
                    "formal_reps": [
                        ["Carl Patel", 19, ["toy car", "toy cars"]],
                        ["Nicholas Morales", -1],
                        ["Jerry Peterson", 1]
                    ],
                    "quantity_ops": [19, -1, 1],
                    "context_list": [
                        "Carl Patel had 19 toy cars in the beginning.",
                        "Carl Patel gave Nicholas Morales 1 toy car.",
                        "Jerry Peterson gave Carl Patel 1 toy car."
                    ],
                    "answer": 19
                },
                {
                    "formal_reps": [
                        ["Carl Patel", 2, [ "toy bear", "toy bears"]],
                        ["Ronald James", 1],
                        ["Clarence Turner", 2]
                    ],
                    "quantity_ops": [2, 1, 2],
                    "context_list": [
                        "Carl Patel had 2 toy bears in the beginning.",
                        "Ronald James gave Carl Patel 1 toy bear.",
                        "Clarence Turner gave Carl Patel 2 toy bears."
                    ],
                    "answer": 5
                },
                {
                    "formal_reps": [
                        ["Paul Cook", 16, ["toy bear", "toy bears"]],
                        ["Ronald Evans", 0],
                        ["Clarence Reed", 2]
                    ],
                    "quantity_ops": [16, 0, 2],
                    "context_list": [
                        "Paul Cook had 16 toy bears in the beginning.",
                        "Ronald Evans did not give Paul Cook any toy bears.",
                        "Clarence Reed gave Paul Cook 2 toy bears."
                    ],
                    "answer": 18
                }
            ],
            "selected_chain_idx": 2,
            "context_string": ("Carl Patel had 19 toy cars in the beginning. "
                               "Carl Patel gave Nicholas Morales 1 toy car. "
                               "Jerry Peterson gave Carl Patel 1 toy car. "
                               "Carl Patel had 2 toy bears in the beginning. "
                               "Ronald James gave Carl Patel 1 toy bear. "
                               "Clarence Turner gave Carl Patel 2 toy bears. "
                               "Paul Cook had 16 toy bears in the beginning. "
                               "Ronald Evans did not give Paul Cook any toy bears. "
                               "Clarence Reed gave Paul Cook 2 toy bears."),
            "question_string": "How many toy bears did Paul Cook have in the end?",
            "answer": 18,
            "depth": 2
        }
    ]

    def test_pattern_gen_prog_depth_0_data(self, debug_flag=False, pattern="generate_program", pattern_num=2):
        evr_instances = GenerateEVRChainingData.generate_evr_data_one_instance(self.test_instances[0])
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check pattern gen prog 1
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a chaining task. "
                     "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                     "episodic_buffer_2: No one exchanged items with others. "
                     "episodic_buffer_3: How many toy bears did Paul Cook have in the end?"),
                    ("#0 = 'According to chunk 0, how many toy bears did Paul Cook have in the beginning?'; "
                     "new_mem(#0);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 2
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, "
                     "how many toy bears did Paul Cook have in the beginning?"),
                    ("#0 = 'chunk_0'; "
                     "#1 = get_chunk(#0); "
                     "#2 = qa(#1, episodic_buffer_0); "
                     "add_to_episodic(#2);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 3
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, "
                     "how many toy bears did Paul Cook have in the beginning? "
                     "episodic_buffer_1: According to chunk 0, Paul Cook had 16 toy bears in the beginning."),
                    "return(episodic_buffer_1);"
                ) in in_out_pairs
        )

        # Check pattern gen prog 4
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a chaining task. "
                     "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                     "episodic_buffer_2: No one exchanged items with others. "
                     "episodic_buffer_3: How many toy bears did Paul Cook have in the end? "
                     "episodic_buffer_4: According to chunk 0, Paul Cook had 16 toy bears in the beginning."),
                    ("#0 = qa(episodic_buffer_4, episodic_buffer_2, episodic_buffer_3); "
                     "return(#0);")
                ) in in_out_pairs
        )

    def test_pattern_gen_prog_depth_2_data(self, debug_flag=False, pattern="generate_program", pattern_num=13):
        evr_instances = GenerateEVRChainingData.generate_evr_data_one_instance(self.test_instances[1])
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check pattern gen prog 1
        assert (
            (
                ("generate_program: "
                 "episodic_buffer_0: This is a chaining task. "
                 "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                 "episodic_buffer_2: Chunk 1 to chunk 3 can be used to infer "
                    "how many items each person had after exchanging. "
                 "episodic_buffer_3: How many toy bears did Paul Cook have in the end?"),
                ("#0 = 'According to chunk 0, how many toy bears did Paul Cook have in the beginning?'; "
                 "new_mem(#0);")
            ) in in_out_pairs
        )

        # Check pattern gen prog 2
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, "
                        "how many toy bears did Paul Cook have in the beginning?"),
                    ("#0 = 'chunk_0'; "
                     "#1 = get_chunk(#0); "
                     "#2 = qa(#1, episodic_buffer_0); "
                     "add_to_episodic(#2);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 3
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, "
                        "how many toy bears did Paul Cook have in the beginning? "
                     "episodic_buffer_1: According to chunk 0, Paul Cook had 16 toy bears in the beginning."),
                    "return(episodic_buffer_1);"
                ) in in_out_pairs
        )

        # Check pattern gen prog 4
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a chaining task. "
                     "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                     "episodic_buffer_2: Chunk 1 to chunk 3 can be used to infer "
                     "how many items each person had after exchanging. "
                     "episodic_buffer_3: How many toy bears did Paul Cook have in the end? "
                     "episodic_buffer_4: According to chunk 0, Paul Cook had 16 toy bears in the beginning."),
                    "clear_mem();"
                ) in in_out_pairs
        )

        # Check pattern gen prog 5
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a chaining task. "
                     "episodic_buffer_1: Chunk 1 to chunk 3 can be used to infer "
                     "how many items each person had after exchanging. "
                     "episodic_buffer_2: How many toy bears did Paul Cook have in the end? "
                     "episodic_buffer_3: According to chunk 0, Paul Cook had 16 toy bears in the beginning."),
                    ("#0 = 'According to the chunks from chunk 1 to chunk 3, "
                        "how many toy bears did Paul Cook have after exchanging?'; "
                     "new_mem(episodic_buffer_3, #0);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 6
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "episodic_buffer_1: According to the chunks from chunk 1 to chunk 3, "
                        "how many toy bears did Paul Cook have after exchanging?"),
                    ("#0 = 'According to the chunks from chunk 1 to chunk 3, "
                        "which chunk can be used to infer how many toy bears Paul Cook had after exchanging?'; "
                     "new_mem(#0);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 7
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to the chunks from chunk 1 to chunk 3, "
                        "which chunk can be used to infer how many toy bears Paul Cook had after exchanging?"),
                    ("#0 = list_chunk_nums('chunk_1', 'chunk_3'); "
                     "for #1 in #0; "
                     "#2 = get_chunk(#1); "
                     "#3 = 'Can this chunk be used to infer how many toy bears Paul Cook had after exchanging?'; "
                     "#4 = qa(#2, #3); "
                     "if #4 == 'True'; "
                     "#5 = #1; "
                     "else; "
                     "pass; "
                     "end_if; "
                     "end_for; "
                     "#6 = rewrite(episodic_buffer_0, #5); "
                     "add_to_episodic(#6);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 8
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to the chunks from chunk 1 to chunk 3, "
                        "which chunk can be used to infer how many toy bears Paul Cook had after exchanging? "
                     "episodic_buffer_1: Chunk 3 can be used to infer "
                        "how many toy bears Paul Cook had after exchanging."),
                    "return(episodic_buffer_1);"
                ) in in_out_pairs
        )

        # Check pattern gen prog 9
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "episodic_buffer_1: According to the chunks from chunk 1 to chunk 3, "
                        "how many toy bears did Paul Cook have after exchanging? "
                     "episodic_buffer_2: Chunk 3 can be used to infer "
                        "how many toy bears Paul Cook had after exchanging."),
                    ("#0 = 'According to chunk 3, how many toy bears did Paul Cook have after exchanging?'; "
                     "new_mem(episodic_buffer_0, #0);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 10
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "episodic_buffer_1: According to chunk 3, "
                        "how many toy bears did Paul Cook have after exchanging?"),
                    ("#0 = 'chunk_3'; "
                     "#1 = 'Paul Cook had 16 toy bears.'; "
                     "while check_next_statement(#0); "
                     "#2 = get_next_statement_num(#0); "
                     "#3 = get_statement(#0, #2); "
                     "#4 = 'How many toy bears did Paul Cook have after exchanging?'; "
                     "#1 = qa(#1, #3, #4); "
                     "end_while; "
                     "add_to_episodic(#1);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 11
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "episodic_buffer_1: According to chunk 3, "
                        "how many toy bears did Paul Cook have after exchanging? "
                     "episodic_buffer_2: Paul Cook had 18 toy bears."),
                    ("#0 = rewrite(episodic_buffer_1, episodic_buffer_2); "
                     "return(#0);")
                ) in in_out_pairs
        )

        # Check pattern gen prog 12
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "episodic_buffer_1: According to the chunks from chunk 1 to chunk 3, "
                     "how many toy bears did Paul Cook have after exchanging? "
                     "episodic_buffer_2: Chunk 3 can be used to infer "
                        "how many toy bears Paul Cook had after exchanging. "
                     "episodic_buffer_3: Paul Cook had 18 toy bears after exchanging."),
                    "return(episodic_buffer_3);"
                ) in in_out_pairs
        )

        # Check pattern gen prog 13
        assert (
                (
                    ("generate_program: "
                     "episodic_buffer_0: This is a chaining task. "
                     "episodic_buffer_1: Chunk 1 to chunk 3 can be used to infer "
                     "how many items each person had after exchanging. "
                     "episodic_buffer_2: How many toy bears did Paul Cook have in the end? "
                     "episodic_buffer_3: According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "episodic_buffer_4: Paul Cook had 18 toy bears after exchanging."),
                    ("#0 = rewrite(episodic_buffer_2, episodic_buffer_4); "
                     "return(#0);")
                ) in in_out_pairs
        )

    def test_pattern_qa_depth_0(self, debug_flag=False, pattern="qa", pattern_num=4):
        evr_instances = GenerateEVRChainingData.generate_evr_data_one_instance(self.test_instances[0])
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test QA pattern 4
        assert (
                (
                    ("qa: "
                     "According to chunk 0, Paul Cook had 16 toy bears in the beginning. "
                     "No one exchanged items with others. "
                     "How many toy bears did Paul Cook have in the end?"),
                    "16"
                ) in in_out_pairs
        )

    def test_pattern_qa_depth_2(self, debug_flag=False, pattern="qa", pattern_num=3):
        evr_instances = GenerateEVRChainingData.generate_evr_data_one_instance(self.test_instances[1])
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check QA pattern 1
        assert (
                (
                    ("qa: "
                     "statement_0: Carl Patel had 19 toy cars in the beginning. "
                     "statement_1: Carl Patel had 2 toy bears in the beginning. "
                     "statement_2: Paul Cook had 16 toy bears in the beginning. "
                     "According to chunk 0, how many toy bears did Paul Cook have in the beginning?"),
                    "According to chunk 0, Paul Cook had 16 toy bears in the beginning."
                ) in in_out_pairs
        )

        # Check QA pattern 2
        assert (
                (
                    ("qa: "
                     "statement_0: Carl Patel gave Nicholas Morales 1 toy car. "
                     "statement_1: Jerry Peterson gave Carl Patel 1 toy car. "
                     "Can this chunk be used to infer how many toy bears Paul Cook had after exchanging?"),
                    "False"
                ) in in_out_pairs
        )

        assert (
                (
                    ("qa: "
                     "statement_0: Ronald James gave Carl Patel 1 toy bear. "
                     "statement_1: Clarence Turner gave Carl Patel 2 toy bears. "
                     "Can this chunk be used to infer how many toy bears Paul Cook had after exchanging?"),
                    "False"
                ) in in_out_pairs
        )

        assert (
                (
                    ("qa: "
                     "statement_0: Ronald Evans did not give Paul Cook any toy bears. "
                     "statement_1: Clarence Reed gave Paul Cook 2 toy bears. "
                     "Can this chunk be used to infer how many toy bears Paul Cook had after exchanging?"),
                    "True"
                ) in in_out_pairs
        )

        # Check QA pattern 3
        assert (
                (
                    ("qa: "
                     "Paul Cook had 16 toy bears. "
                     "Ronald Evans did not give Paul Cook any toy bears. "
                     "How many toy bears did Paul Cook have after exchanging?"),
                    "Paul Cook had 16 toy bears."
                ) in in_out_pairs
        )

        assert (
                (
                    ("qa: "
                     "Paul Cook had 16 toy bears. "
                     "Clarence Reed gave Paul Cook 2 toy bears. "
                     "How many toy bears did Paul Cook have after exchanging?"),
                    "Paul Cook had 18 toy bears."
                ) in in_out_pairs
        )

    def test_pattern_rewrite_depth_2(self, debug_flag=True, pattern="rewrite", pattern_num=3):
        evr_instances = GenerateEVRChainingData.generate_evr_data_one_instance(self.test_instances[1])
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check rewrite pattern 1
        assert (
                (
                    ("rewrite: "
                     "According to the chunks from chunk 1 to chunk 3, "
                        "which chunk can be used to infer how many toy bears Paul Cook had after exchanging? "
                     "3"),
                    "Chunk 3 can be used to infer how many toy bears Paul Cook had after exchanging."
                ) in in_out_pairs
        )

        # Check rewrite pattern 2
        assert (
                (
                    ("rewrite: "
                     "According to chunk 3, how many toy bears did Paul Cook have after exchanging? "
                     "Paul Cook had 18 toy bears."),
                    "Paul Cook had 18 toy bears after exchanging."
                ) in in_out_pairs
        )

        # Check rewrite pattern 3
        assert (
                (
                    ("rewrite: "
                     "How many toy bears did Paul Cook have in the end? "
                     "Paul Cook had 18 toy bears after exchanging."),
                    "18"
                ) in in_out_pairs
        )

    def test_pattern_clear_mem_depth_2(self, debug_flag=False, pattern="clear_mem", pattern_num=1):
        evr_instances = GenerateEVRChainingData.generate_evr_data_one_instance(self.test_instances[1])
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == pattern and e_i["pattern"] == pattern_num:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                (
                    ("clear_mem: "
                     "episodic_buffer_0: This is a chaining task. "
                     "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                     "episodic_buffer_2: Chunk 1 to chunk 3 can be used to infer "
                        "how many items each person had after exchanging. "
                     "episodic_buffer_3: How many toy bears did Paul Cook have in the end? "
                     "episodic_buffer_4: According to chunk 0, Paul Cook had 16 toy bears in the beginning."),
                    ("episodic_buffer_0: 'This is a chaining task.' "
                     "episodic_buffer_1: 'Chunk 1 to chunk 3 can be used to infer "
                        "how many items each person had after exchanging.' "
                     "episodic_buffer_2: 'How many toy bears did Paul Cook have in the end?' "
                     "episodic_buffer_3: 'According to chunk 0, Paul Cook had 16 toy bears in the beginning.'")
                ) in in_out_pairs
        )


if __name__ == "__main__":
    unittest.main()
