import math
import copy

from preliminary_experiments.data_generation.data_base_class import DataBase
from preliminary_experiments.data_generation.data_0_chaining_evr import GenerateEVRChainingData
from preliminary_experiments.data_generation.data_2_tree_search_evr import GenerateEVRTreeSearchData


class GenerateEVRChainingTreeSearchData(DataBase):

    @classmethod
    def get_item_by_quantity(cls, quantity, item_tuple):

        if int(quantity) == 1:
            return item_tuple[0]
        else:
            return item_tuple[1]

    @classmethod
    def formal_statement_to_nl(cls, formal_statement):

        main_chara = formal_statement[0]
        quantity = formal_statement[1]
        item = formal_statement[2]

        item_nl = cls.get_item_by_quantity(quantity, item)

        statement_nl = f"{main_chara} had {quantity} {item_nl}"

        return statement_nl

    @classmethod
    def formal_statement_to_nl_question(cls, formal_statement):

        main_chara = formal_statement[0]
        item = formal_statement[2]

        statement_nl = f"How many {item[1]} did {main_chara} have"

        return statement_nl

    @classmethod
    def split_to_chunks(cls, statements, rules, s_chunk_size=5, r_chunk_size=3):
        num_s_chunks = math.ceil(len(statements) / s_chunk_size)
        statement_chunks = [statements[idx * s_chunk_size: (idx + 1) * s_chunk_size] for idx in range(num_s_chunks)]

        num_r_chunks = math.ceil(len(rules) / r_chunk_size)
        rule_chunks = [rules[idx * r_chunk_size: (idx + 1) * r_chunk_size] for idx in range(num_r_chunks)]

        return statement_chunks, rule_chunks

    @classmethod
    def generate_pattern_gen_prog_1_data(cls, instance):
        evr_instances = []

        num_chains = len(instance["chaining_instance"]["chains"])

        if instance["depth"] == 0:
            input_text_list = [
                "This is a chaining tree search task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                instance["question_string"]
            ]
        else:
            input_text_list = [
                "This is a chaining tree search task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk 1 to chunk {num_chains} can be used to infer how many items each person had after exchanging.",
                instance["question_string"]
            ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        target_text_list = [
            "#0 = 'How many items did each person have after exchanging?';",
            "new_mem(episodic_buffer_1, episodic_buffer_2, #0);",
        ]
        target = " ".join(target_text_list)

        evr_instance = {
            "task": "inter_generate_program",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": -1
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_2_data(cls, instance):
        evr_instances = []

        num_chains = len(instance["chaining_instance"]["chains"])

        if instance["depth"] != 0:
            input_text_list = [
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk 1 to chunk {num_chains} can be used to infer how many items each person had after exchanging.",
                "How many items did each person have after exchanging?"
            ]
        else:
            input_text_list = [
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                "How many items did each person have after exchanging?"
            ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        target_text_list = [
            "#0 = [];",
            "#1 = 'This is a chaining task.';",
            "#2 = 'chunk_0';",
            "while check_next_statement(#2);",
            "#3 = get_next_statement_num(#2);",
            "#4 = get_statement(#2, #3);",
            "#5 = subq(#4, episodic_buffer_2);",
            "#6 = rewrite(episodic_buffer_0, #4);",
            "new_mem(#1, episodic_buffer_0, episodic_buffer_1, #5, #6);",
            "#0 = append_to_list(#0, episodic_buffer_3);",
            "del('episodic_buffer_3');",
            "end_while;",
            "add_to_episodic('#0 stores the number of items each person had after exchanging.');",
        ]
        target = " ".join(target_text_list)

        evr_instance = {
            "task": "inter_generate_program",
            "pattern": 2,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": -1
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_3_data(cls, instance):

        evr_instances = []

        num_chains = len(instance["chaining_instance"]["chains"])

        if instance["depth"] == 0:
            input_text_list = [
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                "How many items did each person have after exchanging?",
                "#0 stores the number of items each person had after exchanging."
            ]
        else:
            input_text_list = [
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk 1 to chunk {num_chains} can be used to infer how many items each person had after exchanging.",
                "How many items did each person have after exchanging?",
                "#0 stores the number of items each person had after exchanging."
            ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        if instance["depth"] != 0:
            target_text_list = [
                "update_chunk('chunk_0', #0);",
                f"#1 = list_chunk_nums('chunk_1', 'chunk_{num_chains}');",
                "for #2 in #1;",
                "del(#2);",
                "end_for;",
                "clean_chunks();",
                "#3 = 'The task is converted to a tree search task.';",
                "return(#3);",
            ]
        else:
            target_text_list = [
                "update_chunk('chunk_0', #0);",
                "clean_chunks();",
                "#1 = 'The task is converted to a tree search task.';",
                "return(#1);"
            ]
        target = " ".join(target_text_list)

        evr_instance = {
            "task": "inter_generate_program",
            "pattern": 3,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": -1
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_4_data(cls, instance):

        evr_instances = []

        num_chains = len(instance["chaining_instance"]["chains"])

        if instance["depth"] == 0:
            input_text_list = [
                "This is a chaining tree search task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                instance["question_string"],
                "The task is converted to a tree search task."
            ]
        else:
            input_text_list = [
                "This is a chaining tree search task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk {1} to chunk {num_chains} can be used to infer how many items each person had after exchanging.",
                instance["question_string"],
                "The task is converted to a tree search task."
            ]

        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        target_text_list = [
            "clear_mem();"
        ]
        target = " ".join(target_text_list)

        evr_instance = {
            "task": "inter_generate_program",
            "pattern": 4,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": -1
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_clear_mem_1_data(cls, instance):

        evr_instances = []

        num_chains = len(instance["chaining_instance"]["chains"])

        if instance["depth"] == 0:
            input_text_list = [
                "This is a chaining tree search task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                instance["question_string"],
                "The task is converted to a tree search task."
            ]
        else:
            input_text_list = [
                "This is a chaining tree search task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk {1} to chunk {num_chains} can be used to infer how many items each person had after exchanging.",
                instance["question_string"],
                "The task is converted to a tree search task."
            ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"clear_mem: {context}"

        target_text_list = [
            "'This is a tree search task.'",
            f"'{instance['question_string']}'"
        ]
        target = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(target_text_list)])

        evr_instance = {
            "task": "inter_clear_mem",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": -1
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_subq_1_data(cls, instance):

        evr_instances = []

        chunk_0_statements = [chain["context_list"][0] for chain in instance["chaining_instance"]["chains"]]

        for ch_idx, chunk_0_s in enumerate(chunk_0_statements):
            main_chara = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][0]
            quant = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][1]
            item_tuple = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][2]

            context = f"{chunk_0_s} How many items did each person have after exchanging?"
            input_text = f"subq: {context}"

            target = f"How many {item_tuple[1]} did {main_chara} have in the end?"

            evr_instance = {
                "task": "inter_subq",
                "pattern": 1,
                "context": context,
                "input": input_text,
                "target": target,
                "org_id": instance["id"],
                "depth": instance["depth"],
                "search_depth": -1
            }

            evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_1_data(cls, instance):

        evr_instances = []

        chunk_0_statements = [chain["context_list"][0] for chain in instance["chaining_instance"]["chains"]]

        for ch_idx, chunk_0_s in enumerate(chunk_0_statements):
            main_chara = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][0]
            quant = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][1]
            item_tuple = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][2]

            context_list = [
                "Chunk 0 answers how many items each person had in the beginning.",
                chunk_0_s
            ]
            context = " ".join(context_list)
            input_text = f"rewrite: {context}"

            target = ("According to chunk 0, "
                      f"{cls.formal_statement_to_nl((main_chara, quant, item_tuple))} in the beginning.")

            evr_instance = {
                "task": "inter_rewrite",
                "pattern": 1,
                "context": context,
                "input": input_text,
                "target": target,
                "org_id": instance["id"],
                "depth": instance["depth"],
                "search_depth": -1
            }

            evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_2_data(cls, instance):

        evr_instances = []

        if instance["depth"] != 0:
            chunk_0_statements = [chain["context_list"][0] for chain in instance["chaining_instance"]["chains"]]

            for ch_idx, chunk_0_s in enumerate(chunk_0_statements):
                main_chara = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][0]
                quant = instance["chaining_instance"]["chains"][ch_idx]["answer"]
                item_tuple = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][2]

                context_list = [
                    f"{cls.formal_statement_to_nl_question((main_chara, quant, item_tuple))} in the end?",
                    f"{cls.formal_statement_to_nl((main_chara, quant, item_tuple))} after exchanging."
                ]
                context = " ".join(context_list)
                input_text = f"rewrite: {context}"

                target = f"{cls.formal_statement_to_nl((main_chara, quant, item_tuple))}."

                evr_instance = {
                    "task": "inter_rewrite",
                    "pattern": 2,
                    "context": context,
                    "input": input_text,
                    "target": target,
                    "org_id": instance["id"],
                    "depth": instance["depth"],
                    "search_depth": -1
                }

                evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_qa_1_data(cls, instance):

        evr_instances = []

        if instance["depth"] == 0:
            chunk_0_statements = [chain["context_list"][0] for chain in instance["chaining_instance"]["chains"]]

            for ch_idx, chunk_0_s in enumerate(chunk_0_statements):
                main_chara = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][0]
                quant = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][1]
                item_tuple = instance["chaining_instance"]["chains"][ch_idx]["formal_reps"][0][2]

                context_list = [
                    (f"According to chunk 0, {cls.formal_statement_to_nl((main_chara, quant, item_tuple))} "
                     "in the beginning."),
                    "No one exchanged items with others.",
                    f"{cls.formal_statement_to_nl_question((main_chara, quant, item_tuple))} in the end?"
                ]
                context = " ".join(context_list)
                input_text = f"qa: {context}"

                target = f"{cls.formal_statement_to_nl((main_chara, quant, item_tuple))}."

                evr_instance = {
                    "task": "inter_qa",
                    "pattern": 1,
                    "context": context,
                    "input": input_text,
                    "target": target,
                    "org_id": instance["id"],
                    "depth": instance["depth"],
                    "search_depth": -1
                }

                evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_evr_data_one_instance_inter(cls, instance):
        evr_instances_gen_funcs = {
            "inter_generate_program": {
                1: cls.generate_pattern_gen_prog_1_data,
                2: cls.generate_pattern_gen_prog_2_data,
                3: cls.generate_pattern_gen_prog_3_data,
                4: cls.generate_pattern_gen_prog_4_data,
            },
            "inter_clear_mem": {
                1: cls.generate_pattern_clear_mem_1_data,
            },
            "inter_subq": {
                1: cls.generate_pattern_subq_1_data,
            },
            "inter_rewrite": {
                1: cls.generate_pattern_rewrite_1_data,
                2: cls.generate_pattern_rewrite_2_data,
            },
            "inter_qa": {
                1: cls.generate_pattern_qa_1_data
            }
        }

        evr_instances_all_patterns = []
        for data_p in evr_instances_gen_funcs:
            for data_n in evr_instances_gen_funcs[data_p]:
                evr_instances_all_patterns.extend(evr_instances_gen_funcs[data_p][data_n](instance))

        return evr_instances_all_patterns

    @classmethod
    def generate_evr_data_one_instance_chaining(cls, instance):

        pattern_generation_func = {
            "qa": {
                2: GenerateEVRChainingData.generate_pattern_qa_2_data,
                3: GenerateEVRChainingData.generate_pattern_qa_3_data,
            },

            "rewrite": {
                1: GenerateEVRChainingData.generate_pattern_rewrite_1_data,
                2: GenerateEVRChainingData.generate_pattern_rewrite_2_data,
            },

            "gen_prog": {
                4: GenerateEVRChainingData.generate_pattern_gen_prog_4_data,
                5: GenerateEVRChainingData.generate_pattern_gen_prog_5_data,
                6: GenerateEVRChainingData.generate_pattern_gen_prog_6_data,
                7: GenerateEVRChainingData.generate_pattern_gen_prog_7_data,
                8: GenerateEVRChainingData.generate_pattern_gen_prog_8_data,
                9: GenerateEVRChainingData.generate_pattern_gen_prog_9_data,
                10: GenerateEVRChainingData.generate_pattern_gen_prog_10_data,
                11: GenerateEVRChainingData.generate_pattern_gen_prog_11_data,
                12: GenerateEVRChainingData.generate_pattern_gen_prog_12_data,
                13: GenerateEVRChainingData.generate_pattern_gen_prog_13_data,
            },

            "clear_mem": {
                1: GenerateEVRChainingData.generate_pattern_clear_mem_data
            }
        }

        evr_instances_chaining = []

        for chain_idx, chain in enumerate(instance["chaining_instance"]["chains"]):
            for pattern in pattern_generation_func.keys():
                for pattern_num in pattern_generation_func[pattern].keys():
                    evr_instances = pattern_generation_func[pattern][pattern_num](
                        instance["chaining_instance"], chain_idx, chain)

                    evr_instances_chaining.extend(evr_instances)

        for evr_in in evr_instances_chaining:
            evr_in["task"] = f"chaining_{evr_in['task']}"

        return evr_instances_chaining

    @classmethod
    def generate_evr_data_one_instance_tree_search(cls, instance):
        tree_search_instance_for_evr = copy.deepcopy(instance["tree_search_instance"])
        tree_search_instance_for_evr["statements"]["grounded"][0] = instance["initial_s_grounded"]
        tree_search_instance_for_evr["statement_indices_shuffle_map"] = {
            str(i): i for i in range(len(instance["initial_s_grounded"]))}
        (evr_instances_tree_search, query_proved_flag, proof_chk_idx, statement_nl_chunks, rule_nl_chunks,
         traversal_history, depth_history) = GenerateEVRTreeSearchData.generate_evr_data_one_instance(
            tree_search_instance_for_evr)
        for evr_in in evr_instances_tree_search:
            evr_in["task"] = f"tree_search_{evr_in['task']}"

        return evr_instances_tree_search

    @classmethod
    def generate_evr_data_one_instance(cls, instance):

        instance["chaining_instance"]["id"] = "-"
        instance["tree_search_instance"]["id"] = "-"

        # Generate and process the chaining evr data
        evr_instances_chaining = cls.generate_evr_data_one_instance_chaining(instance)

        # Generate and process the tree search data
        evr_instances_tree_search = cls.generate_evr_data_one_instance_tree_search(instance)

        # Generate and proces the intermediate data
        evr_instances_inter = cls.generate_evr_data_one_instance_inter(instance)

        cartesian_tree_search_instances = []
        cartesian_tree_search_instances.extend(evr_instances_chaining)
        cartesian_tree_search_instances.extend(evr_instances_inter)
        cartesian_tree_search_instances.extend(evr_instances_tree_search)

        return cartesian_tree_search_instances

    @classmethod
    def generate_evr_instances(cls, instances):
        evr_instances_all = []
        for instance in instances:
            evr_instances = cls.generate_evr_data_one_instance(instance)

            evr_instances_all.extend(evr_instances)

        return evr_instances_all

    @classmethod
    def get_evr_chunks(cls, instance, rule_chunk_size=3):
        # First add the chaining statements to the chunk
        external_chunks = [[chain["context_list"][0] for chain in instance["chaining_instance"]["chains"]]]
        if instance["depth"] >= 1:
            external_chunks.extend([chain["context_list"][1:] for chain in instance["chaining_instance"]["chains"]])

        # Then add the tree search rules to the chunk
        tree_search_rules = [r for r in instance["tree_search_instance"]["context_list"] if r.startswith("If")]
        if len(tree_search_rules) > 0:
            n_r_chunk = math.ceil(len(tree_search_rules) / rule_chunk_size)
            for r_c_idx in range(n_r_chunk):
                external_chunks.append(tree_search_rules[r_c_idx * rule_chunk_size: (r_c_idx + 1) * rule_chunk_size])

        return external_chunks

    @classmethod
    def generate_evr_eval_instances(cls, instances):
        for instance in instances:
            external_chunks = cls.get_evr_chunks(instance)
            instance["external_chunks"] = {
                "chunk_" + str(chunk_idx): {
                    "statement_" + str(statement_idx): statement
                    for statement_idx, statement in enumerate(chunk)
                } for chunk_idx, chunk in enumerate(external_chunks)
            }

            # Prepare the episodic buffer
            num_chains = len(instance["chaining_instance"]["chains"])

            if instance["depth"] == 0:
                instance["episodic_buffer_dict"] = {
                    "episodic_buffer_0": "This is a chaining tree search task.",
                    "episodic_buffer_1": "Chunk 0 answers how many items each person had in the beginning.",
                    "episodic_buffer_2": "No one exchanged items with others.",
                    "episodic_buffer_3": f"{instance['question_string']}",
                }
            else:
                instance["episodic_buffer_dict"] = {
                    "episodic_buffer_0": "This is a chaining tree search task.",
                    "episodic_buffer_1": "Chunk 0 answers how many items each person had in the beginning.",
                    "episodic_buffer_2": (f"Chunk 1 to chunk {num_chains} can be used to infer "
                                          "how many items each person had after exchanging."),
                    "episodic_buffer_3": f"{instance['question_string']}",
                }

        return instances


if __name__ == "__main__":
    pass
