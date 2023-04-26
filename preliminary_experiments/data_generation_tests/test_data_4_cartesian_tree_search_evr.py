import unittest

from preliminary_experiments.data_generation.data_4_cartesian_tree_search_evr import GenerateEVRCartesianTreeSearchData


class TestGenerateCartesianTreeSearchData(unittest.TestCase):

    test_instance = {
        'cartesian_instance': {
            'id': '8453484e0b3ca74b9f438071dfba6d8a',
            'depth': 2,
            'context_string': 'Each of Sean Sanchez and Christopher King had 9 owls and 4 rulers.',
            'question_string': 'List the items and the number of each item each person had.',
            'answer': 'Sean Sanchez had 9 owls, Sean Sanchez had 4 rulers, Christopher King had 9 owls, Christopher King had 4 rulers.',
            'target_list': [
                ('Sean Sanchez', 9, ('owl', 'owls')),
                ('Sean Sanchez', 4, ('ruler', 'rulers')),
                ('Christopher King', 9, ('owl', 'owls')),
                ('Christopher King', 4, ('ruler', 'rulers'))
            ],
            'target_nl_list': [
                'Sean Sanchez had 9 owls', 'Sean Sanchez had 4 rulers', 'Christopher King had 9 owls',
                'Christopher King had 4 rulers'
            ],
            'ungrounded_list': [
                ('Andrew Flores', 4, ('ruler', 'rulers')),
                ('Christopher King', 4, ('apple', 'apples')),
                ('Fred Price', 20, ('apple', 'apples')),
                ('Harold Murphy', 8, ('ruler', 'rulers'))
            ],
            'ungrounded_nl_list': [
                'Andrew Flores had 4 rulers', 'Christopher King had 4 apples', 'Fred Price had 20 apples',
                'Harold Murphy had 8 rulers'
            ],
            'target_len': 33
        },
        'tree_search_instance': {
            'depth': 1,
            'provable': 1,
            'statements': {
                'grounded': [
                    [('Sean Sanchez', 9, ('owl', 'owls')), ('Christopher King', 4, ('ruler', 'rulers'))],
                    [('Henry Thompson', 17, ('banana', 'bananas')), ('Joshua Adams', 18, ('banana', 'bananas'))]
                ],
                'ungrounded': [
                    [('Andrew Flores', 4, ('ruler', 'rulers')), ('Fred Price', 20, ('apple', 'apples'))],
                    [('Henry Thompson', 14, ('toy car', 'toy cars')), ('Frank Rivera', 10, ('banana', 'bananas'))]
                ],
                'distractors': [[]]
            },
            'rules': {
                'grounded 1 var': [
                    [((('Christopher King', 4, ('ruler', 'rulers')),), ('Henry Thompson', 17, ('banana', 'bananas')))]
                ],
                'grounded 2 var': [
                    [((('Christopher King', 4, ('ruler', 'rulers')), ('Sean Sanchez', 9, ('owl', 'owls'))),
                      ('Joshua Adams', 18, ('banana', 'bananas')))]
                ],
                'ungrounded 1 var': [
                    [((('Andrew Flores', 4, ('ruler', 'rulers')),), ('Henry Thompson', 14, ('toy car', 'toy cars'))),
                     ((('Fred Price', 20, ('apple', 'apples')),), ('Frank Rivera', 10, ('banana', 'bananas')))]
                ], 'ungrounded 2 var': [[]], 'backtracking': [None]
            },
            'question': ('Henry Thompson', 17, ('banana', 'bananas')),
            'answer': 'Yes',
            'context_list': [
                'Sean Sanchez had 9 owls.', 'Christopher King had 4 rulers.',
                'If Christopher King had 4 rulers then Henry Thompson had 17 bananas.',
                'If Christopher King had 4 rulers and Sean Sanchez had 9 owls then Joshua Adams had 18 bananas.',
                'If Fred Price had 20 apples then Frank Rivera had 10 bananas.',
                'If Andrew Flores had 4 rulers then Henry Thompson had 14 toy cars.'
            ],
            'context_string': ('Sean Sanchez had 9 owls. Christopher King had 4 rulers. '
                               'If Christopher King had 4 rulers then Henry Thompson had 17 bananas. '
                               'If Christopher King had 4 rulers and Sean Sanchez had 9 owls then Joshua Adams had 18 bananas. '
                               'If Fred Price had 20 apples then Frank Rivera had 10 bananas. '
                               'If Andrew Flores had 4 rulers then Henry Thompson had 14 toy cars.'),
            'question_string': 'Did Henry Thompson have 17 bananas?',
            'statement_indices_shuffle_map': {0: 0, 1: 1},
            'rule_indices_shuffle_map': {0: 0, 1: 1, 3: 2, 2: 3},
            'target_text_w_inter': ('If Christopher King had 4 rulers then Henry Thompson had 17 bananas. '
                                    'Christopher King had 4 rulers. answer: Yes')
        },
        'answer': 'Yes',
        'depth': 1,
        'context_string': ('Each of Sean Sanchez and Christopher King had 9 owls and 4 rulers. '
                           'If Christopher King had 4 rulers then Henry Thompson had 17 bananas. '
                           'If Christopher King had 4 rulers and Sean Sanchez had 9 owls then Joshua Adams had 18 bananas. '
                           'If Fred Price had 20 apples then Frank Rivera had 10 bananas. '
                           'If Andrew Flores had 4 rulers then Henry Thompson had 14 toy cars.'),
        'question_string': 'Did Henry Thompson have 17 bananas?',
        'target_text': 'Yes',
        'target_text_w_inter': ('Sean Sanchez had 9 owls. Sean Sanchez had 4 rulers. '
                                'Christopher King had 9 owls. Christopher King had 4 rulers. '
                                'If Christopher King had 4 rulers then Henry Thompson had 17 bananas. '
                                'Christopher King had 4 rulers. answer: Yes'),
        "id": "some_id"
    }

    def test_generate_program(self, debug_flag=False):

        evr_instances = GenerateEVRCartesianTreeSearchData.generate_evr_data_one_instance(self.test_instance)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "inter_generate_program" and e_i["pattern"] == 3:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test generate_program_1
        assert (
            ("generate_program: "
             "episodic_buffer_0: This is a cartesian tree search task. "
             "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
             "episodic_buffer_2: Did Henry Thompson have 17 bananas?"),
            ("#0 = 'This is a cartesian task.'; "
             "#1 = 'List the items that each person had.'; "
             "new_mem(#0, episodic_buffer_1, #1);")
        ) in in_out_pairs

        # Test generate_program_2
        assert (
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
                    "update_chunk('chunk_0', #2); "
                    "clean_chunks(); "
                    "return('The task is converted to a tree search task.');")
               ) in in_out_pairs

        # Test generate_program_3
        assert (
                   ("generate_program: "
                    "episodic_buffer_0: This is a cartesian tree search task. "
                    "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                    "episodic_buffer_2: Did Henry Thompson have 17 bananas? "
                    "episodic_buffer_3: The task is converted to a tree search task."),
                   "clear_mem();"
               ) in in_out_pairs

    def test_clear_mem(self, debug_flag=True):

        evr_instances = GenerateEVRCartesianTreeSearchData.generate_evr_data_one_instance(self.test_instance)
        in_out_pairs = [(e_i["input"], e_i["target"]) for e_i in evr_instances]

        if debug_flag:
            print("=" * 40)
            for e_i in evr_instances:
                if e_i["task"] == "inter_clear_mem" and e_i["pattern"] == 1:
                    print(e_i["input"])
                    print("\t", e_i["target"])

        # Test clear_mem
        assert (
                   ("clear_mem: "
                    "episodic_buffer_0: This is a cartesian tree search task. "
                    "episodic_buffer_1: Chunk 0 can be used to infer the number of items each person had. "
                    "episodic_buffer_2: Did Henry Thompson have 17 bananas? "
                    "episodic_buffer_3: The task is converted to a tree search task."),
                   ("episodic_buffer_0: 'This is a tree search task.' "
                    "episodic_buffer_1: 'Did Henry Thompson have 17 bananas?'")
               ) in in_out_pairs


if __name__ == "__main__":
    unittest.main()