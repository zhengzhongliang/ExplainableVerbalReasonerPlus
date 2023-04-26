import unittest

from preliminary_experiments.data_generation.data_3_chaining_tree_search_evr import GenerateEVRChainingTreeSearchData


class TestGenerateChainingTreeSearchData(unittest.TestCase):

    test_instance_d2 = {
      "chaining_instance": {
        "id": "de787ac2062a1570de93ac988f2e74b2",
        "chains": [
          {
            "formal_reps": [
              ["Kenneth White", 10, ["pear", "pears"]],
              ["Charles Jackson", -2], ["Frank Ross", 1]
            ],
            "quantity_ops": [10, -2, 1],
            "context_list": [
              "Kenneth White had 10 pears in the beginning.",
              "Kenneth White gave Charles Jackson 2 pears.",
              "Frank Ross gave Kenneth White 1 pear."
            ],
            "answer": 9
          },
          {
            "formal_reps": [
              ["Andrew Castillo", 18, ["pen", "pens"]],
              ["Raymond Martin", 1], ["Clarence Taylor", 1]
            ],
            "quantity_ops": [18, 1, 1],
            "context_list": [
              "Andrew Castillo had 18 pens in the beginning.",
              "Raymond Martin gave Andrew Castillo 1 pen.",
              "Clarence Taylor gave Andrew Castillo 1 pen."
            ],
            "answer": 20
          },
          {
            "formal_reps": [
              ["Brandon Hill", 10, ["peach", "peaches"]],
              ["Eric Kim", 0], ["Benjamin Cooper", 1]],
            "quantity_ops": [10, 0, 1],
            "context_list": [
              "Brandon Hill had 10 peaches in the beginning.",
              "Eric Kim did not give Brandon Hill any peaches.",
              "Benjamin Cooper gave Brandon Hill 1 peach."
            ],
            "answer": 11
          },
          {
            "formal_reps": [
              ["Brandon Hill", 3, ["kitten", "kittens"]],
              ["Nicholas Turner", 0], ["Joseph Baker", 0]
            ],
            "quantity_ops": [3, 0, 0],
            "context_list": [
              "Brandon Hill had 3 kittens in the beginning.",
              "Nicholas Turner did not give Brandon Hill any kittens.",
              "Joseph Baker did not give Brandon Hill any kittens."
            ],
            "answer": 3
          },
          {
            "formal_reps": [
              ["Christopher Torres", 1, ["pen", "pens"]],
              ["Edward Adams", 2], ["Eric Campbell", 1]
            ],
            "quantity_ops": [1, 2, 1],
            "context_list": [
              "Christopher Torres had 1 pen in the beginning.",
              "Edward Adams gave Christopher Torres 2 pens.",
              "Eric Campbell gave Christopher Torres 1 pen."
            ],
            "answer": 4
          },
          {
            "formal_reps": [
              ["Brian Campbell", 10, ["kitten", "kittens"]],
              ["Douglas Stewart", 0], ["Gregory Kim", 1]
            ],
            "quantity_ops": [10, 0, 1],
            "context_list": [
              "Brian Campbell had 10 kittens in the beginning.",
              "Douglas Stewart did not give Brian Campbell any kittens.",
              "Gregory Kim gave Brian Campbell 1 kitten."
            ],
            "answer": 11
          }
        ],
        "selected_chain_idx": 5,
        "context_string": "Kenneth White had 10 pears in the beginning. Kenneth White gave Charles Jackson 2 pears. Frank Ross gave Kenneth White 1 pear. Andrew Castillo had 18 pens in the beginning. Raymond Martin gave Andrew Castillo 1 pen. Clarence Taylor gave Andrew Castillo 1 pen. Brandon Hill had 10 peaches in the beginning. Eric Kim did not give Brandon Hill any peaches. Benjamin Cooper gave Brandon Hill 1 peach. Brandon Hill had 3 kittens in the beginning. Nicholas Turner did not give Brandon Hill any kittens. Joseph Baker did not give Brandon Hill any kittens. Christopher Torres had 1 pen in the beginning. Edward Adams gave Christopher Torres 2 pens. Eric Campbell gave Christopher Torres 1 pen. Brian Campbell had 10 kittens in the beginning. Douglas Stewart did not give Brian Campbell any kittens. Gregory Kim gave Brian Campbell 1 kitten.",
        "question_string": "How many kittens did Brian Campbell have in the end?",
        "answer": 11,
        "depth": 2
      },
      "tree_search_instance": {
        "depth": 2,
        "provable": 0,
        "statements": {
          "grounded": [
            [
              ["Brandon Hill", 11, ["peach", "peaches"]],
              ["Brian Campbell", 11, ["kitten", "kittens"]],
              ["Christopher Torres", 4, ["pen", "pens"]]
            ],
            [
              ["Joe Kim", 5, ["ruler", "rulers"]],
              ["Stephen Foster", 14, ["banana", "bananas"]]
            ],
            [
              ["Howard Thompson", 3, ["apple", "apples"]],
              ["Anthony Martinez", 6, ["apple", "apples"]]
            ]
          ],
          "ungrounded": [
            [
              ["Kenneth White", 1, ["pear", "pears"]],
              ["Andrew Castillo", 2, ["pen", "pens"]],
              ["Brandon Hill", 7, ["kitten", "kittens"]]
            ],
            [
              ["Shawn Scott", 18, ["owl", "owls"]],
              ["Gregory Cox", 13, ["apple", "apples"]]
            ],
            [
              ["Howard Thompson", 2, ["apple", "apples"]],
              ["Scott Wright", 3, ["apple", "apples"]]
            ]
          ],
          "distractors": [
            [],
            []
          ]
        },
        "rules": {
          "grounded 1 var": [
            [
              [[["Brandon Hill", 11, ["peach", "peaches"]]],
                ["Joe Kim", 5, ["ruler", "rulers"]]]
            ],
            [
              [[["Joe Kim", 5, ["ruler", "rulers"]]],
                ["Howard Thompson", 3, ["apple", "apples"]]]
            ]
          ],
          "grounded 2 var": [
            [
              [[["Christopher Torres", 4, ["pen", "pens"]], ["Brian Campbell", 11, ["kitten", "kittens"]]],
                ["Stephen Foster", 14, ["banana", "bananas"]]]
            ],
            [
              [[["Joe Kim", 5, ["ruler", "rulers"]], ["Stephen Foster", 14, ["banana", "bananas"]]],
                ["Anthony Martinez", 6, ["apple", "apples"]]]
            ]
          ],
          "ungrounded 1 var": [
            [
              [[["Brandon Hill", 7, ["kitten", "kittens"]]], ["Shawn Scott", 18, ["owl", "owls"]]]
            ],
            [
              [[["Gregory Cox", 13, ["apple", "apples"]]], ["Howard Thompson", 2, ["apple", "apples"]]]
            ]
          ],
          "ungrounded 2 var": [
            [
              [[["Brandon Hill", 7, ["kitten", "kittens"]], ["Andrew Castillo", 2, ["pen", "pens"]]],
                ["Gregory Cox", 13, ["apple", "apples"]]]
            ],
            [
              [[["Gregory Cox", 13, ["apple", "apples"]], ["Shawn Scott", 18, ["owl", "owls"]]],
                ["Scott Wright", 3, ["apple", "apples"]]]
            ]
          ],
          "backtracking": [
            [
              [["Brandon Hill", 11, ["peach", "peaches"]], ["Brandon Hill", 7, ["kitten", "kittens"]]],
              ["Stephen Foster", 14, ["banana", "bananas"]]
            ],
            [
              [["Gregory Cox", 13, ["apple", "apples"]]], ["Anthony Martinez", 6, ["apple", "apples"]]
            ]
          ]
        },
        "question": [
          "Scott Wright",
          3,
          [
            "apple",
            "apples"
          ]
        ],
        "answer": "No",
        "context_list": [
          "Brian Campbell had 11 kittens.",
          "Brandon Hill had 11 peaches.",
          "Christopher Torres had 4 pens.",
          "If Brandon Hill had 11 peaches then Joe Kim had 5 rulers.",
          "If Brandon Hill had 7 kittens then Shawn Scott had 18 owls.",
          "If Brandon Hill had 7 kittens and Andrew Castillo had 2 pens then Gregory Cox had 13 apples.",
          "If Gregory Cox had 13 apples then Anthony Martinez had 6 apples.",
          "If Gregory Cox had 13 apples and Shawn Scott had 18 owls then Scott Wright had 3 apples.",
          "If Joe Kim had 5 rulers then Howard Thompson had 3 apples.",
          "If Joe Kim had 5 rulers and Stephen Foster had 14 bananas then Anthony Martinez had 6 apples.",
          "If Brandon Hill had 11 peaches and Brandon Hill had 7 kittens then Stephen Foster had 14 bananas.",
          "If Gregory Cox had 13 apples then Howard Thompson had 2 apples.",
          "If Christopher Torres had 4 pens and Brian Campbell had 11 kittens then Stephen Foster had 14 bananas."
        ],
        "context_string": "Brian Campbell had 11 kittens. Brandon Hill had 11 peaches. Christopher Torres had 4 pens. If Brandon Hill had 11 peaches then Joe Kim had 5 rulers. If Brandon Hill had 7 kittens then Shawn Scott had 18 owls. If Brandon Hill had 7 kittens and Andrew Castillo had 2 pens then Gregory Cox had 13 apples. If Gregory Cox had 13 apples then Anthony Martinez had 6 apples. If Gregory Cox had 13 apples and Shawn Scott had 18 owls then Scott Wright had 3 apples. If Joe Kim had 5 rulers then Howard Thompson had 3 apples. If Joe Kim had 5 rulers and Stephen Foster had 14 bananas then Anthony Martinez had 6 apples. If Brandon Hill had 11 peaches and Brandon Hill had 7 kittens then Stephen Foster had 14 bananas. If Gregory Cox had 13 apples then Howard Thompson had 2 apples. If Christopher Torres had 4 pens and Brian Campbell had 11 kittens then Stephen Foster had 14 bananas.",
        "question_string": "Did Scott Wright have 3 apples?",
        "statement_indices_shuffle_map": {
          "1": 0,
          "0": 1,
          "2": 2
        },
        "rule_indices_shuffle_map": {
          "0": 0, "4": 1, "6": 2, "9": 3, "7": 4, "1": 5, "3": 6,
          "8": 7, "5": 8, "2": 9
        },
        "target_text_w_inter": "If Gregory Cox had 13 apples and Shawn Scott had 18 owls then Scott Wright had 3 apples. If Brandon Hill had 7 kittens and Andrew Castillo had 2 pens then Gregory Cox had 13 apples. answer: No"
      },
      "initial_s_grounded": [
        ["Kenneth White", 9, ["pear", "pears"]],
        ["Andrew Castillo", 20, ["pen", "pens"]],
        ["Brandon Hill", 11, ["peach", "peaches"]],
        ["Brandon Hill", 3, ["kitten", "kittens"]],
        ["Christopher Torres", 4, ["pen", "pens"]],
        ["Brian Campbell", 11, ["kitten", "kittens"]]
      ],
      "initial_s_grounded_selected": [
        ["Brandon Hill", 11, ["peach", "peaches"]],
        ["Brian Campbell", 11, ["kitten", "kittens"]],
        ["Christopher Torres", 4, ["pen", "pens"]]
      ],
      "initial_s_ungrounded": [
        ["Kenneth White", 1, ["pear", "pears"]],
        ["Andrew Castillo", 2, ["pen", "pens"]],
        ["Brandon Hill", 7, ["kitten", "kittens"]]
      ],
      "id": "some_id",
      "depth": 2,
      "context_string": "Kenneth White had 10 pears in the beginning. Kenneth White gave Charles Jackson 2 pears. Frank Ross gave Kenneth White 1 pear. Andrew Castillo had 18 pens in the beginning. Raymond Martin gave Andrew Castillo 1 pen. Clarence Taylor gave Andrew Castillo 1 pen. Brandon Hill had 10 peaches in the beginning. Eric Kim did not give Brandon Hill any peaches. Benjamin Cooper gave Brandon Hill 1 peach. Brandon Hill had 3 kittens in the beginning. Nicholas Turner did not give Brandon Hill any kittens. Joseph Baker did not give Brandon Hill any kittens. Christopher Torres had 1 pen in the beginning. Edward Adams gave Christopher Torres 2 pens. Eric Campbell gave Christopher Torres 1 pen. Brian Campbell had 10 kittens in the beginning. Douglas Stewart did not give Brian Campbell any kittens. Gregory Kim gave Brian Campbell 1 kitten. If Brandon Hill had 11 peaches then Joe Kim had 5 rulers. If Brandon Hill had 7 kittens then Shawn Scott had 18 owls. If Brandon Hill had 7 kittens and Andrew Castillo had 2 pens then Gregory Cox had 13 apples. If Gregory Cox had 13 apples then Anthony Martinez had 6 apples. If Gregory Cox had 13 apples and Shawn Scott had 18 owls then Scott Wright had 3 apples. If Joe Kim had 5 rulers then Howard Thompson had 3 apples. If Joe Kim had 5 rulers and Stephen Foster had 14 bananas then Anthony Martinez had 6 apples. If Brandon Hill had 11 peaches and Brandon Hill had 7 kittens then Stephen Foster had 14 bananas. If Gregory Cox had 13 apples then Howard Thompson had 2 apples. If Christopher Torres had 4 pens and Brian Campbell had 11 kittens then Stephen Foster had 14 bananas.",
      "question_string": "Did Scott Wright have 3 apples?",
      "target_text": "No",
      "target_text_w_inter": "Kenneth White had 9 pears. Andrew Castillo had 20 pens. Brandon Hill had 11 peaches. Brandon Hill had 3 kittens. Christopher Torres had 4 pens. Brian Campbell had 11 kittens. If Gregory Cox had 13 apples and Shawn Scott had 18 owls then Scott Wright had 3 apples. If Brandon Hill had 7 kittens and Andrew Castillo had 2 pens then Gregory Cox had 13 apples. answer: No"
    }

    test_instance_d0 = {
        "chaining_instance": {
            "id": "045d9f93ab14097afd530a7e342e3824",
            "chains": [
                {
                    "formal_reps": [
                        ["Jack Long", 6, ["kitten", "kittens"]]
                    ],
                    "quantity_ops": [6],
                    "context_list": [
                        "Jack Long had 6 kittens in the beginning."
                    ],
                    "answer": 6
                },
                {
                    "formal_reps": [
                        ["Steven Gutierrez", 1, ["puppy", "puppies"]]
                    ],
                    "quantity_ops": [1],
                    "context_list": [
                        "Steven Gutierrez had 1 puppy in the beginning."
                    ],
                    "answer": 1
                },
                {
                    "formal_reps": [
                        ["Kevin Ortiz", 1, ["ruler", "rulers"]]
                    ],
                    "quantity_ops": [1],
                    "context_list": [
                        "Kevin Ortiz had 1 ruler in the beginning."
                    ],
                    "answer": 1
                },
                {
                    "formal_reps": [
                        ["Thomas Hughes", 14, ["puppy", "puppies"]]
                    ],
                    "quantity_ops": [14],
                    "context_list": [
                        "Thomas Hughes had 14 puppies in the beginning."
                    ],
                    "answer": 14
                }
            ],
            "selected_chain_idx": 0,
            "context_string": "Jack Long had 6 kittens in the beginning. Steven Gutierrez had 1 puppy in the beginning. Kevin Ortiz had 1 ruler in the beginning. Thomas Hughes had 14 puppies in the beginning.",
            "question_string": "How many kittens did Jack Long have in the end?",
            "answer": 6,
            "depth": 0
        },
        "tree_search_instance": {
            "depth": 0,
            "provable": 1,
            "statements": {
                "grounded": [
                    [["Thomas Hughes", 14, ["puppy", "puppies"]], ["Jack Long", 6, ["kitten", "kittens"]]]
                ],
                "ungrounded": [
                    [["Steven Gutierrez", 4, ["puppy", "puppies"]], ["Kevin Ortiz", 4, ["ruler", "rulers"]]]
                ],
                "distractors": []
            },
            "rules": {
                "grounded 1 var": [],
                "grounded 2 var": [],
                "ungrounded 1 var": [],
                "ungrounded 2 var": [],
                "backtracking": []
            },
            "question": ["Thomas Hughes", 14, ["puppy", "puppies"]],
            "answer": "Yes",
            "context_list": ["Jack Long had 6 kittens.", "Thomas Hughes had 14 puppies."],
            "context_string": "Jack Long had 6 kittens. Thomas Hughes had 14 puppies.",
            "question_string": "Did Thomas Hughes have 14 puppies?",
            "statement_indices_shuffle_map": {"1": 0, "0": 1},
            "rule_indices_shuffle_map": {},
            "target_text_w_inter": "Thomas Hughes had 14 puppies. answer: Yes"
        },
        "initial_s_grounded": [
            ["Jack Long", 6, ["kitten", "kittens"]], ["Steven Gutierrez", 1, ["puppy", "puppies"]],
            ["Kevin Ortiz", 1, ["ruler", "rulers"]], ["Thomas Hughes", 14, ["puppy", "puppies"]]
        ],
        "initial_s_grounded_selected": [
            ["Thomas Hughes", 14, ["puppy", "puppies"]], ["Jack Long", 6, ["kitten", "kittens"]]
        ],
        "initial_s_ungrounded": [
            ["Steven Gutierrez", 4, ["puppy", "puppies"]], ["Kevin Ortiz", 4, ["ruler", "rulers"]]
        ],
        "id": "some_id",
        "depth": 0,
        "context_string": "Jack Long had 6 kittens in the beginning. Steven Gutierrez had 1 puppy in the beginning. Kevin Ortiz had 1 ruler in the beginning. Thomas Hughes had 14 puppies in the beginning. ", "question_string": "Did Thomas Hughes have 14 puppies?",
        "target_text": "Yes",
        "target_text_w_inter": "Jack Long had 6 kittens. Steven Gutierrez had 1 puppy. Kevin Ortiz had 1 ruler. Thomas Hughes had 14 puppies. Thomas Hughes had 14 puppies. answer: Yes"
    }

    def test_depth_0_data(self, debug_flag=False):

        evr_instances = GenerateEVRChainingTreeSearchData.generate_evr_data_one_instance(self.test_instance_d0)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "inter_qa" and e_i["pattern"] == 1:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check generate_program pattern 1
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a chaining tree search task. "
                    "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_2: No one exchanged items with others. "
                    "episodic_buffer_3: Did Thomas Hughes have 14 puppies?"),
                   ("#0 = 'How many items did each person have after exchanging?'; "
                    "new_mem(episodic_buffer_1, episodic_buffer_2, #0);")
               ) in in_out_pairs

        # Check generate_program pattern 2
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_1: No one exchanged items with others. "
                    "episodic_buffer_2: How many items did each person have after exchanging?"),
                   ("#0 = []; "
                    "#1 = 'This is a chaining task.'; "
                    "#2 = 'chunk_0'; "
                    "while check_next_statement(#2); "
                    "#3 = get_next_statement_num(#2); "
                    "#4 = get_statement(#2, #3); "
                    "#5 = subq(#4, episodic_buffer_2); "
                    "#6 = rewrite(episodic_buffer_0, #4); "
                    "new_mem(#1, episodic_buffer_0, episodic_buffer_1, #5, #6); "
                    "#0 = append_to_list(#0, episodic_buffer_3); "
                    "del('episodic_buffer_3'); "
                    "end_while; "
                    "add_to_episodic('#0 stores the number of items each person had after exchanging.');")
               ) in in_out_pairs

        # Check generate_program chaining 4
        assert (
            ("generate_program: "
             "episodic_buffer_0: This is a chaining task. "
             "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
             "episodic_buffer_2: No one exchanged items with others. "
             "episodic_buffer_3: How many puppies did Thomas Hughes have in the end? "
             "episodic_buffer_4: According to chunk 0, Thomas Hughes had 14 puppies in the beginning."),
            ("#0 = qa(episodic_buffer_4, episodic_buffer_2, episodic_buffer_3); "
             "return(#0);")
        ) in in_out_pairs

        # Check generate_program 3
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_1: No one exchanged items with others. "
                    "episodic_buffer_2: How many items did each person have after exchanging? "
                    "episodic_buffer_3: #0 stores the number of items each person had after exchanging."),
                   ("update_chunk('chunk_0', #0); "
                    "clean_chunks(); "
                    "#1 = 'The task is converted to a tree search task.'; "
                    "return(#1);")
               ) in in_out_pairs

        # Check generate_program 4
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a chaining tree search task. "
                    "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_2: No one exchanged items with others. "
                    "episodic_buffer_3: Did Thomas Hughes have 14 puppies? "
                    "episodic_buffer_4: The task is converted to a tree search task."),
                   "clear_mem();"
               ) in in_out_pairs

        # Check qa 1
        assert (
            ("qa: "
             "According to chunk 0, Thomas Hughes had 14 puppies in the beginning. "
             "No one exchanged items with others. "
             "How many puppies did Thomas Hughes have in the end?"),
            "Thomas Hughes had 14 puppies."
        ) in in_out_pairs

    def test_depth_2_data(self, debug_flag=True):

        evr_instances = GenerateEVRChainingTreeSearchData.generate_evr_data_one_instance(self.test_instance_d2)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "tree_search_qa" and e_i["pattern"] == 1:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Check generate_program pattern 1
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a chaining tree search task. "
                    "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_2: Chunk 1 to chunk 6 can be used to infer "
                        "how many items each person had after exchanging. "
                    "episodic_buffer_3: Did Scott Wright have 3 apples?"),
                   ("#0 = 'How many items did each person have after exchanging?'; "
                    "new_mem(episodic_buffer_1, episodic_buffer_2, #0);")
               ) in in_out_pairs

        # Check generate_program pattern 2
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_1: Chunk 1 to chunk 6 can be used to infer "
                        "how many items each person had after exchanging. "
                    "episodic_buffer_2: How many items did each person have after exchanging?"),
                   ("#0 = []; "
                    "#1 = 'This is a chaining task.'; "
                    "#2 = 'chunk_0'; "
                    "while check_next_statement(#2); "
                    "#3 = get_next_statement_num(#2); "
                    "#4 = get_statement(#2, #3); "
                    "#5 = subq(#4, episodic_buffer_2); "
                    "#6 = rewrite(episodic_buffer_0, #4); "
                    "new_mem(#1, episodic_buffer_0, episodic_buffer_1, #5, #6); "
                    "#0 = append_to_list(#0, episodic_buffer_3); "
                    "del('episodic_buffer_3'); "
                    "end_while; "
                    "add_to_episodic('#0 stores the number of items each person had after exchanging.');")
               ) in in_out_pairs

        # Check generate_program chaining 4
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a chaining task. "
                    "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_2: Chunk 1 to chunk 6 can be used to infer "
                        "how many items each person had after exchanging. "
                    "episodic_buffer_3: How many pears did Kenneth White have in the end? "
                    "episodic_buffer_4: According to chunk 0, Kenneth White had 10 pears in the beginning."),
                   "clear_mem();"
               ) in in_out_pairs

        # Check generate_program 3
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_1: Chunk 1 to chunk 6 can be used to infer "
                        "how many items each person had after exchanging. "
                    "episodic_buffer_2: How many items did each person have after exchanging? "
                    "episodic_buffer_3: #0 stores the number of items each person had after exchanging."),
                   ("update_chunk('chunk_0', #0); "
                    "#1 = list_chunk_nums('chunk_1', 'chunk_6'); "
                    "for #2 in #1; "
                    "del(#2); " 
                    "end_for; "
                    "clean_chunks(); "
                    "#3 = 'The task is converted to a tree search task.'; "
                    "return(#3);")
               ) in in_out_pairs

        # Check generate_program 4
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a chaining tree search task. "
                    "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_2: Chunk 1 to chunk 6 can be used to infer "
                        "how many items each person had after exchanging. "
                    "episodic_buffer_3: Did Scott Wright have 3 apples? "
                    "episodic_buffer_4: The task is converted to a tree search task."),
                   "clear_mem();"
               ) in in_out_pairs

        # Check subq 1
        assert (
            ("subq: "
             "Kenneth White had 10 pears in the beginning. "
             "How many items did each person have after exchanging?"),
            "How many pears did Kenneth White have in the end?"
        ) in in_out_pairs

        # Check rewrite 1
        assert (
                   ("rewrite: "
                    "Chunk 0 answers how many items each person had in the beginning. "
                    "Kenneth White had 10 pears in the beginning."),
                   "According to chunk 0, Kenneth White had 10 pears in the beginning."
               ) in in_out_pairs

        # Check rewrite 2
        assert (
            ("rewrite: "
             "How many pears did Kenneth White have in the end? "
             "Kenneth White had 9 pears after exchanging."),
            "Kenneth White had 9 pears."
        ) in in_out_pairs

        # Check clear mem
        assert (
                   ("clear_mem: "
                    "episodic_buffer_0: This is a chaining tree search task. "
                    "episodic_buffer_1: Chunk 0 answers how many items each person had in the beginning. "
                    "episodic_buffer_2: Chunk 1 to chunk 6 can be used to infer "
                        "how many items each person had after exchanging. "
                    "episodic_buffer_3: Did Scott Wright have 3 apples? "
                    "episodic_buffer_4: The task is converted to a tree search task."),
                   ("episodic_buffer_0: 'This is a tree search task.' "
                    "episodic_buffer_1: 'Did Scott Wright have 3 apples?'"
                    )
               ) in in_out_pairs

        # Check tree search qa
        # assert (
        #     (),
        #     ()
        # ) in in_out_pairs


if __name__ == "__main__":
    unittest.main()
