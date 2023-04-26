import unittest

from preliminary_experiments.data_generation.data_2_tree_search import GenerateTreeSearchData
from preliminary_experiments.data_generation_debug.debug_data_2_tree_search import DebugGenerateTreeSearchData

from preliminary_experiments.data_generation.data_2_tree_search_evr import GenerateEVRTreeSearchData


class TestGenerateTreeSearchData(unittest.TestCase):
    """

    What are needed to test the evr data generation:
    statements, rules, statement_indices_shuffle_map, rule_indices_shuffle_map,
    context_list, question
    """

    def test_pattern_gen_prog_1_n_3(self, debug_flag=False):
        test_instance = {
            "statements": {
                "grounded": [
                    [
                        ("Steve James", 10, ("puppy", "puppies")),
                        ("Dennis White", 6, ("ruler", "rulers")),
                        ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                    ]
                ],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((("Steve James", 10, ("puppy", "puppies")), ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))),
                     ('Robert Rogers', 14, ('puppy', 'puppies')))
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "context_list": [
                "Steve James had 10 puppies.",
                "Dennis White had 6 rulers.",
                "Nicholas Mendoza had 10 rabbits.",
                "If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then Robert Rogers had 14 puppies.",
            ],
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": "1",
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and (e_i["pattern"] == 1 or e_i["pattern"] == 3):
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check generate_program pattern 1
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a tree search task. "
                    "episodic_buffer_1: Did Robert Rogers have 14 puppies?"),
                   ("#0 = 'Which chunk can prove Robert Rogers had 14 puppies?'; "
                    "new_mem(#0); ")
               ) in in_out_pairs

        # Check generate_program pattern 3
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 0 be used to prove Robert Rogers had 14 puppies?"),
                   ("#0 = get_chunk('chunk_0'); "
                    "#1 = qa(#0, episodic_buffer_0); "
                    "add_to_episodic(#1); ")
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies?"),
                   ("#0 = get_chunk('chunk_1'); "
                    "#1 = qa(#0, episodic_buffer_0); "
                    "add_to_episodic(#1); ")
               ) in in_out_pairs

    def test_pattern_gen_prog_2(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [
                    [
                        ("Steve James", 10, ("puppy", "puppies")),
                        ("Dennis White", 6, ("ruler", "rulers")),
                        ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                    ]
                ],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((("Steve James", 10, ("puppy", "puppies")), ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))),
                     ('Robert Rogers', 14, ('puppy', 'puppies')))
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "context_list": [
                "Steve James had 10 puppies.",
                "Dennis White had 6 rulers.",
                "Nicholas Mendoza had 10 rabbits.",
                "If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then Robert Rogers had 14 puppies.",
            ],
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": "1",
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 2:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies?"),
                   ("while check_next_chunk(); "
                    "#0 = get_next_chunk_num(); "
                    f"#1 = 'Can this chunk prove Robert Rogers had 14 puppies?'; "
                    "#2 = rewrite(#1, #0); "
                    "new_mem(#2); "
                    "end_while; ")
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: Chunk 0 can not prove Robert Rogers had 14 puppies. "
                    "episodic_buffer_2: Chunk 1 can prove Robert Rogers had 14 puppies."),
                   "clear_mem(); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ("Steve James", 11, ("puppy", "puppies"))
        test_instance["context_list"][0] = "Steve James had 11 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: Chunk 0 can not prove Robert Rogers had 14 puppies. "
                    "episodic_buffer_2: Chunk 1 can not prove Robert Rogers had 14 puppies."),
                   "clear_mem(); "
               ) in in_out_pairs

    def test_pattern_gen_prog_4_1_1_n_4_1_2(self, debug_flag=False):

        test_instance = {"statements": {
            "grounded": [
                [
                    ("Steve James", 10, ("puppy", "puppies")),
                    ("Dennis White", 6, ("ruler", "rulers")),
                    ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                ]
            ],
            "distractors": [[]],
        }, "rules": {
            "grounded 1 var": [[
                ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
            ]],
            "grounded 2 var": [[]],
            "ungrounded 1 var": [[]],
            "ungrounded 2 var": [[]],
            "backtracking": [[]]
        }, "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2}, "rule_indices_shuffle_map": {0: 0}, "context_list": [
            "Steve James had 10 puppies.",
            "Dennis White had 6 rulers",
            "Nicholas Mendoza had 10 rabbits",
            "If Eugene Williams had 17 rulers then William Phillips had 17 kittens.",
        ], "question": ("Steve James", 10, ("puppy", "puppies")), "id": "1", "depth": 2}

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                ("generate_program: "
                 "episodic_buffer_0: Can chunk 0 be used to prove Steve James had 10 puppies? "
                 "episodic_buffer_1: Chunk 0 can prove Steve James had 10 puppies."),
                "return(episodic_buffer_1); "
               ) in in_out_pairs

        test_instance["question"] = ("Steve James", 9, ("puppy", "puppies"))

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 0 be used to prove Steve James had 9 puppies? "
                    "episodic_buffer_1: Chunk 0 can not prove Steve James had 9 puppies."),
                   "return(episodic_buffer_1); "
               ) in in_out_pairs

    def test_pattern_gen_prog_4_2_1(self, debug_flag=False):
        test_instance = {
            "statements": {
                "grounded": [[
                    ('Eugene Williams', 17, ('ruler', 'rulers'))
                ]],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((('Eugene Williams', 17, ('ruler', 'rulers')),), ('William Phillips', 17, ('kitten', 'kittens'))),
                    ((('Steve James', 9, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "context_list": [
                "Eugene Williams had 17 rulers.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Eugene Williams had 17 rulers then William Phillips had 17 kittens.",
                "If Steve James had 9 puppies then Robert Rogers had 14 puppies.",
            ],
            "statement_indices_shuffle_map": {0: 0},
            "rule_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "question": ('William Phillips', 17, ('kitten', 'kittens')),
            "id": 1,
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove William Phillips had 17 kittens? "
                    "episodic_buffer_1: I need to prove Eugene Williams had 17 rulers."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove William Phillips had 17 kittens? "
                    "episodic_buffer_1: I need to prove Eugene Williams had 17 rulers. "
                    "episodic_buffer_2: Eugene Williams had 17 rulers is True."),
                   "#0 = 'Chunk 1 can prove William Phillips had 17 kittens.'; return(#0); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ('Eugene Williams', 16, ('ruler', 'rulers'))
        test_instance["context_list"][0] = "Eugene Williams had 16 rulers."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("-" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove William Phillips had 17 kittens? "
                    "episodic_buffer_1: I need to prove Eugene Williams had 17 rulers."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove William Phillips had 17 kittens? "
                    "episodic_buffer_1: I need to prove Eugene Williams had 17 rulers. "
                    "episodic_buffer_2: Eugene Williams had 17 rulers is not True."),
                   "#0 = 'Chunk 1 can not prove William Phillips had 17 kittens.'; return(#0); "
               ) in in_out_pairs

    def test_pattern_gen_prog_4_2_2(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [[
                    ('Dennis White', 6, ('ruler', 'rulers')),
                    ('Steve James', 10, ('puppy', 'puppies'))
                ]],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((('Eugene Williams', 17, ('ruler', 'rulers')),), ('William Phillips', 17, ('kitten', 'kittens'))),
                    ((('Dennis White', 6, ('ruler', 'rulers')), ('Steve James', 10, ('puppy', 'puppies'))),
                     ('Kenneth James', 14, ('puppy', 'puppies'))),
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "context_list": [
                "Dennis White had 6 rulers.",
                "Steve James had 10 puppies.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Eugene Williams had 17 rulers then William Phillips had 17 kittens.",
                "If Dennis White had 6 rulers and Steve James had 10 puppies then Kenneth James had 14 puppies.",
            ],
            "statement_indices_shuffle_map": {0: 0, 1: 1},
            "rule_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "question": ('Kenneth James', 14, ('puppy', 'puppies')),
            "id": 1,
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Kenneth James had 14 puppies? "
                    "episodic_buffer_1: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Kenneth James had 14 puppies? "
                    "episodic_buffer_1: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_2: Dennis White had 6 rulers and Steve James had 10 puppies is True."),
                   "#0 = 'Chunk 1 can prove Kenneth James had 14 puppies.'; return(#0); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ('Dennis White', 7, ('ruler', 'rulers'))
        test_instance["context_list"][0] = "Dennis White had 7 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("-" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Kenneth James had 14 puppies? "
                    "episodic_buffer_1: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Kenneth James had 14 puppies? "
                    "episodic_buffer_1: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_2: Dennis White had 6 rulers and Steve James had 10 puppies is not True."),
                   "#0 = 'Chunk 1 can not prove Kenneth James had 14 puppies.'; return(#0); "
               ) in in_out_pairs

    def test_pattern_gen_prog_4_2_3(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [[
                    ('Charles Jackson', 2, ('puppy', 'puppies')),
                    ('Eugene Williams', 17, ('ruler', 'rulers'))
                ]],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((('Eugene Williams', 17, ('ruler', 'rulers')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "context_list": [
                "Charles Jackson had 2 puppies.",
                "Eugene Williams had 17 rulers.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Eugene Williams had 17 rulers then Robert Rogers had 14 puppies.",
            ],
            "statement_indices_shuffle_map": {0: 0, 1: 1},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": 1,
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Eugene Williams had 17 rulers."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Eugene Williams had 17 rulers. "
                    "episodic_buffer_2: Charles Jackson had 2 puppies is True. "
                    "episodic_buffer_3: Eugene Williams had 17 rulers is True."),
                   "#0 = 'Chunk 1 can prove Robert Rogers had 14 puppies.'; return(#0); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ('Charles Jackson', 3, ('puppy', 'puppies'))
        test_instance["context_list"][0] = "Charles Jackson had 3 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("-" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Eugene Williams had 17 rulers."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Eugene Williams had 17 rulers. "
                    "episodic_buffer_2: Charles Jackson had 2 puppies is not True. "
                    "episodic_buffer_3: Eugene Williams had 17 rulers is True."),
                   "#0 = 'Chunk 1 can prove Robert Rogers had 14 puppies.'; return(#0); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][1] = ('Eugene Williams', 18, ('ruler', 'rulers'))
        test_instance["context_list"][1] = "Eugene Williams had 18 rulers."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("-" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Eugene Williams had 17 rulers."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Eugene Williams had 17 rulers. "
                    "episodic_buffer_2: Charles Jackson had 2 puppies is not True. "
                    "episodic_buffer_3: Eugene Williams had 17 rulers is not True."),
                   "#0 = 'Chunk 1 can not prove Robert Rogers had 14 puppies.'; return(#0); "
               ) in in_out_pairs

    def test_pattern_gen_prog_4_2_4(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [[
                    ('Charles Jackson', 2, ('puppy', 'puppies')),
                    ('Dennis White', 6, ('ruler', 'rulers')),
                    ('Steve James', 10, ('puppy', 'puppies'))
                ]],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((('Dennis White', 6, ('ruler', 'rulers')), ('Steve James', 10, ('puppy', 'puppies'))),
                     ('Robert Rogers', 14, ('puppy', 'puppies'))),
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "context_list": [
                "Charles Jackson had 2 puppies.",
                "Dennis White had 6 rulers.",
                "Steve James had 10 puppies.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Dennis White had 6 rulers and Steve James had 10 puppies then Robert Rogers had 14 puppies.",
            ],
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": 1,
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Dennis White had 6 rulers and Steve James had 10 puppies."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_2: Charles Jackson had 2 puppies is True. "
                    "episodic_buffer_3: Dennis White had 6 rulers and Steve James had 10 puppies is True."),
                   "#0 = 'Chunk 1 can prove Robert Rogers had 14 puppies.'; return(#0); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ('Charles Jackson', 3, ('puppy', 'puppies'))
        test_instance["context_list"][0] = "Charles Jackson had 3 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("-" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Dennis White had 6 rulers and Steve James had 10 puppies."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_2: Charles Jackson had 2 puppies is not True. "
                    "episodic_buffer_3: Dennis White had 6 rulers and Steve James had 10 puppies is True."),
                   "#0 = 'Chunk 1 can prove Robert Rogers had 14 puppies.'; return(#0); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][1] = ('Dennis White', 7, ('ruler', 'rulers'))
        test_instance["context_list"][1] = "Dennis White had 7 rulers."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("-" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Dennis White had 6 rulers and Steve James had 10 puppies."),
                   "#0 = subqs(); for #1 in #0; new_mem(#1); end_for; "
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_2: Charles Jackson had 2 puppies is not True. "
                    "episodic_buffer_3: Dennis White had 6 rulers and Steve James had 10 puppies is not True."),
                   "#0 = 'Chunk 1 can not prove Robert Rogers had 14 puppies.'; return(#0); "
               ) in in_out_pairs

    def test_pattern_gen_prog_4_2_6(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [[
                    ('Charles Jackson', 2, ('puppy', 'puppies')),
                    ('Dennis White', 6, ('ruler', 'rulers')),
                    ('Steve James', 10, ('puppy', 'puppies'))
                ]],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((('Dennis White', 6, ('ruler', 'rulers')), ('Steve James', 10, ('puppy', 'puppies'))),
                     ('Robert Rogers', 14, ('puppy', 'puppies'))),
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "context_list": [
                "Charles Jackson had 2 puppies.",
                "Dennis White had 6 rulers.",
                "Steve James had 10 puppies.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Dennis White had 6 rulers and Steve James had 10 puppies then Robert Rogers had 14 puppies.",
            ],
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "question": ('Charles Jackson', 3, ('puppy', 'puppies')),
            "id": 1,
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Charles Jackson had 3 puppies? "
                    "episodic_buffer_1: Chunk 1 can not prove Charles Jackson had 3 puppies."),
                   "return(episodic_buffer_1); "
               ) in in_out_pairs

    def test_pattern_gen_prog_5(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [[
                    ('Charles Jackson', 2, ('puppy', 'puppies')),
                    ('Dennis White', 6, ('ruler', 'rulers')),
                    ('Steve James', 10, ('puppy', 'puppies'))
                ]],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((('Dennis White', 6, ('ruler', 'rulers')), ('Steve James', 10, ('puppy', 'puppies'))),
                     ('Robert Rogers', 14, ('puppy', 'puppies'))),
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "context_list": [
                "Charles Jackson had 2 puppies.",
                "Dennis White had 6 rulers.",
                "Steve James had 10 puppies.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Dennis White had 6 rulers and Steve James had 10 puppies then Robert Rogers had 14 puppies.",
            ],
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": 1,
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 5:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Charles Jackson had 2 puppies."),
                   ("#0 = 'This is a tree search task.'; "
                    "#1 = subqs(); "
                    "for #2 in #1; "
                    "new_mem(#0, #2); "
                    "end_for; ")
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Charles Jackson had 2 puppies. "
                    "episodic_buffer_1: Chunk 0 can prove Charles Jackson had 2 puppies."),
                   ("#0 = 'Charles Jackson had 2 puppies is True.'; "
                    "return(#0); ")
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies."),
                   ("#0 = 'This is a tree search task.'; "
                    "#1 = subqs(); "
                    "for #2 in #1; "
                    "new_mem(#0, #2); "
                    "end_for; ")
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_1: Chunk 0 can prove Dennis White had 6 rulers. "
                    "episodic_buffer_2: Chunk 0 can prove Steve James had 10 puppies."),
                   ("#0 = 'Dennis White had 6 rulers and Steve James had 10 puppies is True.'; "
                    "return(#0); ")
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ('Charles Jackson', 3, ('puppy', 'puppies'))
        test_instance["context_list"][0] = "Charles Jackson had 3 puppies."

        test_instance["statements"]["grounded"][0][1] = ('Dennis White', 7, ('ruler', 'rulers'))
        test_instance["context_list"][1] = "Dennis White had 7 rulers."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Charles Jackson had 2 puppies. "
                    "episodic_buffer_1: No chunks can prove Charles Jackson had 2 puppies."),
                   ("#0 = 'Charles Jackson had 2 puppies is not True.'; "
                    "return(#0); ")
               ) in in_out_pairs

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_1: No chunks can prove Dennis White had 6 rulers. "
                    "episodic_buffer_2: Chunk 0 can prove Steve James had 10 puppies."),
                   ("#0 = 'Dennis White had 6 rulers and Steve James had 10 puppies is not True.'; "
                    "return(#0); ")
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][2] = ('Steve James', 11, ('puppy', 'puppies'))
        test_instance["context_list"][2] = "Steve James had 11 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and e_i["pattern"] == 4:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("generate_program: "
                    "episodic_buffer_0: I need to prove Dennis White had 6 rulers and Steve James had 10 puppies. "
                    "episodic_buffer_1: No chunks can prove Dennis White had 6 rulers. "
                    "episodic_buffer_2: No chunks can prove Steve James had 10 puppies."),
                   ("#0 = 'Dennis White had 6 rulers and Steve James had 10 puppies is not True.'; "
                    "return(#0); ")
               ) in in_out_pairs

    def test_pattern_gen_prog_6_n_7(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [
                    [
                        ("Steve James", 10, ("puppy", "puppies")),
                        ("Dennis White", 6, ("ruler", "rulers")),
                        ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                    ]
                ],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((("Steve James", 10, ("puppy", "puppies")), ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))),
                     ('Robert Rogers', 14, ('puppy', 'puppies')))
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "context_list": [
                "Steve James had 10 puppies.",
                "Dennis White had 6 rulers.",
                "Nicholas Mendoza had 10 rabbits.",
                "If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then Robert Rogers had 14 puppies.",
            ],
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": "1",
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "generate_program" and (e_i["pattern"] == 6 or e_i["pattern"] == 7):
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test pattern 6
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: Chunk 1 can prove Robert Rogers had 14 puppies."),
                   "return(episodic_buffer_1); "
               ) in in_out_pairs

        # Test pattern 7
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a tree search task. "
                    "episodic_buffer_1: Did Robert Rogers have 14 puppies? "
                    "episodic_buffer_2: Chunk 1 can prove Robert Rogers had 14 puppies."),
                   "return(episodic_buffer_2); "
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ("Steve James", 11, ("puppy", "puppies"))
        test_instance["context_list"][0] = "Steve James had 11 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        # Test pattern 6
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: No chunks can prove Robert Rogers had 14 puppies."),
                   "return(episodic_buffer_1); "
               ) in in_out_pairs

        # Test pattern 7
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a tree search task. "
                    "episodic_buffer_1: Did Robert Rogers have 14 puppies? "
                    "episodic_buffer_2: No chunks can prove Robert Rogers had 14 puppies."),
                   "return(episodic_buffer_2); "
               ) in in_out_pairs

    def test_pattern_qa_n_rewrite(self, debug_flag=False):

        test_instance = {
            "statements": {
                "grounded": [
                    [
                        ("Steve James", 10, ("puppy", "puppies")),
                        ("Dennis White", 6, ("ruler", "rulers")),
                        ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                    ]
                ],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((("Steve James", 10, ("puppy", "puppies")), ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))),
                     ('Robert Rogers', 14, ('puppy', 'puppies')))
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "context_list": [
                "Steve James had 10 puppies.",
                "Dennis White had 6 rulers.",
                "Nicholas Mendoza had 10 rabbits.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then Robert Rogers had 14 puppies.",
            ],
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": "1",
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "qa" or e_i["task"] == "rewrite":
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test rewrite data
        assert (
                   ("rewrite: "
                    "Can this chunk prove Robert Rogers had 14 puppies? "
                    "chunk_0"),
                   "Can chunk 0 be used to prove Robert Rogers had 14 puppies?"
               ) in in_out_pairs

        assert (
                   ("rewrite: "
                    "Can this chunk prove Robert Rogers had 14 puppies? "
                    "chunk_1"),
                   "Can chunk 1 be used to prove Robert Rogers had 14 puppies?"
               ) in in_out_pairs

        # Test qa data
        assert (
                   ("qa: "
                    "statement_0: Steve James had 10 puppies. "
                    "statement_1: Dennis White had 6 rulers. "
                    "statement_2: Nicholas Mendoza had 10 rabbits. "
                    "Can chunk 0 be used to prove Robert Rogers had 14 puppies?"),
                   "Chunk 0 can not prove Robert Rogers had 14 puppies."
               ) in in_out_pairs

        assert (
                   ("qa: "
                    "statement_0: If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies. "
                    "statement_1: If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then "
                    "Robert Rogers had 14 puppies. "
                    "Can chunk 1 be used to prove Robert Rogers had 14 puppies?"),
                   ("I need to prove Charles Jackson had 2 puppies, or "
                    "Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits.")
               ) in in_out_pairs

    def test_pattern_clear_mem(self, debug_flag=False):

        # Test clear_mem data
        # Need to test it in different conditions, either 1 chunk can prove it, or no chunks can prove it.

        test_instance = {
            "statements": {
                "grounded": [
                    [
                        ("Steve James", 10, ("puppy", "puppies")),
                        ("Dennis White", 6, ("ruler", "rulers")),
                        ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                    ]
                ],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((("Steve James", 10, ("puppy", "puppies")), ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))),
                     ('Robert Rogers', 14, ('puppy', 'puppies')))
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "context_list": [
                "Steve James had 10 puppies.",
                "Dennis White had 6 rulers.",
                "Nicholas Mendoza had 10 rabbits.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then Robert Rogers had 14 puppies.",
            ],
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": "1",
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "clear_mem":
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test clear_mem data
        assert (
                   ("clear_mem: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: Chunk 0 can not prove Robert Rogers had 14 puppies. "
                    "episodic_buffer_2: Chunk 1 can prove Robert Rogers had 14 puppies."),
                   ("episodic_buffer_0: 'Which chunk can prove Robert Rogers had 14 puppies?' "
                    "episodic_buffer_1: 'Chunk 1 can prove Robert Rogers had 14 puppies.'")
               ) in in_out_pairs

        assert (
                   ("clear_mem: "
                    "episodic_buffer_0: Which chunk can prove Steve James had 10 puppies? "
                    "episodic_buffer_1: Chunk 0 can prove Steve James had 10 puppies. "
                    "episodic_buffer_2: Chunk 1 can not prove Steve James had 10 puppies."),
                   ("episodic_buffer_0: 'Which chunk can prove Steve James had 10 puppies?' "
                    "episodic_buffer_1: 'Chunk 0 can prove Steve James had 10 puppies.'")
               ) in in_out_pairs

        test_instance["statements"]["grounded"][0][0] = ("Steve James", 11, ("puppy", "puppies"))
        test_instance["context_list"][0] = "Steve James had 11 puppies."

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "clear_mem":
                    print(e_i["input"])
                    print("\t", e_i["target"])

        assert (
                   ("clear_mem: "
                    "episodic_buffer_0: Which chunk can prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: Chunk 0 can not prove Robert Rogers had 14 puppies. "
                    "episodic_buffer_2: Chunk 1 can not prove Robert Rogers had 14 puppies."),
                   ("episodic_buffer_0: 'Which chunk can prove Robert Rogers had 14 puppies?' "
                    "episodic_buffer_1: 'No chunks can prove Robert Rogers had 14 puppies.'")
               ) in in_out_pairs

    def test_pattern_subqs(self, debug_flag=False):

        # Test subqs data
        # Need to test it in different conditions, test [A], [A and B], [A or B], [A and B, or C]

        test_instance = {
            "statements": {
                "grounded": [
                    [
                        ('Charles Jackson', 2, ('puppy', 'puppies')),
                        ("Steve James", 10, ("puppy", "puppies")),
                        ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))
                    ]
                ],
                "distractors": [[]],
            },
            "rules": {
                "grounded 1 var": [[
                    ((('Charles Jackson', 2, ('puppy', 'puppies')),), ('Robert Rogers', 14, ('puppy', 'puppies'))),
                    ((("Steve James", 10, ("puppy", "puppies")), ("Nicholas Mendoza", 10, ("rabbit", "rabbits"))),
                     ('Robert Rogers', 14, ('puppy', 'puppies')))
                ]],
                "grounded 2 var": [[]],
                "ungrounded 1 var": [[]],
                "ungrounded 2 var": [[]],
                "backtracking": [[]]
            },
            "statement_indices_shuffle_map": {0: 0, 1: 1, 2: 2},
            "rule_indices_shuffle_map": {0: 0, 1: 1},
            "context_list": [
                "Charles Jackson had 2 puppies.",
                "Steve James had 10 puppies.",
                "Nicholas Mendoza had 10 rabbits.",
                "If Charles Jackson had 2 puppies then Robert Rogers had 14 puppies.",
                "If Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits then Robert Rogers had 14 puppies.",
            ],
            "question": ('Robert Rogers', 14, ('puppy', 'puppies')),
            "id": "1",
            "depth": 2
        }

        evr_instances = GenerateEVRTreeSearchData.generate_evr_data_one_instance(test_instance)[0]
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "subqs":
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test subqs 1
        assert (
                   ("subqs: "
                    "episodic_buffer_0: Can chunk 1 be used to prove Robert Rogers had 14 puppies? "
                    "episodic_buffer_1: I need to prove Charles Jackson had 2 puppies, or "
                    "Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits."),
                   ("['I need to prove Charles Jackson had 2 puppies.', "
                    "'I need to prove Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits.']")
               ) in in_out_pairs

        # Test subqs 2
        assert (
                   ("subqs: "
                    "episodic_buffer_0: I need to prove Charles Jackson had 2 puppies."),
                   "['Did Charles Jackson have 2 puppies?']"
               ) in in_out_pairs

        assert (
                   ("subqs: "
                    "episodic_buffer_0: I need to prove Steve James had 10 puppies and Nicholas Mendoza had 10 rabbits."),
                   "['Did Steve James have 10 puppies?', 'Did Nicholas Mendoza have 10 rabbits?']"
               ) in in_out_pairs


if __name__ == "__main__":
    unittest.main()
