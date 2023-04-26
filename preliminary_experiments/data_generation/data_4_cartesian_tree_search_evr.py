import math
import copy

from preliminary_experiments.data_generation.data_base_class import DataBase
from preliminary_experiments.data_generation.data_1_cartesian_evr import GenerateEVRCartesianData
from preliminary_experiments.data_generation.data_2_tree_search_evr import GenerateEVRTreeSearchData


class GenerateEVRCartesianTreeSearchData(DataBase):

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
        quant = formal_statement[1]
        item = formal_statement[2]

        statement_nl = (f"Did {main_chara} have {quant} {item[0]}" if quant == 1
                        else f"Did {main_chara} have {quant} {item[1]}")

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

        question_formal = instance["tree_search_instance"]["question"]

        input_text_list = [
            "This is a cartesian tree search task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            f"{cls.formal_statement_to_nl_question(question_formal)}?"
        ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        target_text_list = [
            "#0 = 'This is a cartesian task.';",
            "#1 = 'List the items that each person had.';",
            "new_mem(#0, episodic_buffer_1, #1);"
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

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_2_data(cls, instance):

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            "#0 stores the list of persons.",
            "#1 stores the list of items."
        ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        target_text_list = [
            "#2 = [];",
            "for #3 in #0;",
            "for #4 in #1;",
            "#5 = 'How many items did this person have?';",
            "#6 = rewrite(#5, #3, #4);",
            "#2 = append_to_list(#2, #6);",
            "end_for;",
            "end_for;",
            "update_chunk('chunk_0', #2);",
            "clean_chunks();",
            "return('The task is converted to a tree search task.');",
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

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_3_data(cls, instance):

        question_formal = instance["tree_search_instance"]["question"]

        input_text_list = [
            "This is a cartesian tree search task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            f"{cls.formal_statement_to_nl_question(question_formal)}?",
            "The task is converted to a tree search task."
        ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"generate_program: {context}"

        target_text_list = [
            "clear_mem();",
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

        return evr_instance

    @classmethod
    def generate_pattern_clear_mem_1_data(cls, instance):

        question_formal = instance["tree_search_instance"]["question"]

        input_text_list = [
            "This is a cartesian tree search task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            f"{cls.formal_statement_to_nl_question(question_formal)}?",
            "The task is converted to a tree search task."
        ]
        context = " ".join([f"episodic_buffer_{idx}: {ep}" for idx, ep in enumerate(input_text_list)])
        input_text = f"clear_mem: {context}"

        target_text_list = [
            "'This is a tree search task.'",
            f"'{cls.formal_statement_to_nl_question(question_formal)}?'",
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

        return evr_instance

    @classmethod
    def generate_evr_data_one_instance_inter(cls, instance):
        """Generate the conversion patterns from cartesian to tree search."""

        evr_inter_instances = []
        evr_inter_instances.append(cls.generate_pattern_gen_prog_1_data(instance))
        evr_inter_instances.append(cls.generate_pattern_gen_prog_2_data(instance))
        evr_inter_instances.append(cls.generate_pattern_gen_prog_3_data(instance))
        evr_inter_instances.append(cls.generate_pattern_clear_mem_1_data(instance))

        return evr_inter_instances

    @classmethod
    def generate_evr_data_one_instance(cls, instance):

        instance["cartesian_instance"]["id"] = "-"
        instance["tree_search_instance"]["id"] = "-"

        # Generate and process the chaining evr data
        evr_instances_cartesian = GenerateEVRCartesianData.generate_evr_data_one_instance(instance["cartesian_instance"])
        evr_instances_cartesian = [evr_in for evr_in in evr_instances_cartesian
                                   if not (evr_in["task"] == "generate_program" and evr_in["pattern"] == 11) and
                                   not (evr_in["task"] == "generate_program" and evr_in["pattern"] == 12)]
        for evr_in in evr_instances_cartesian:
            evr_in["task"] = f"cartesian_{evr_in['task']}"

        # Generate and process the tree search data
        tree_search_instance_for_evr = copy.deepcopy(instance["tree_search_instance"])
        tree_search_instance_for_evr["statements"]["grounded"][0] = instance["cartesian_instance"]["target_list"]
        tree_search_instance_for_evr["statement_indices_shuffle_map"] = {
            str(i): i for i in range(len(instance["cartesian_instance"]["target_list"]))}
        (evr_instances_tree_search, query_proved_flag, proof_chk_idx, statement_nl_chunks, rule_nl_chunks,
         traversal_history, depth_history) = GenerateEVRTreeSearchData.generate_evr_data_one_instance(
            tree_search_instance_for_evr)
        for evr_in in evr_instances_tree_search:
            evr_in["task"] = f"tree_search_{evr_in['task']}"
        evr_instances_inter = cls.generate_evr_data_one_instance_inter(instance)

        cartesian_tree_search_instances = []
        cartesian_tree_search_instances.extend(evr_instances_cartesian)
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
        external_chunks = [[instance["cartesian_instance"]["context_string"]]]

        # Then add the tree search rules to the chunk
        n_initial_s_tree_search = len(instance["tree_search_instance"]["statements"]["grounded"][0])
        tree_search_rules = instance["tree_search_instance"]["context_list"][n_initial_s_tree_search:]
        if len(tree_search_rules) > 0:
            n_r_chunk = math.ceil(len(tree_search_rules) / rule_chunk_size)
            for r_c_idx in range(n_r_chunk):
                external_chunks.append(tree_search_rules[r_c_idx * rule_chunk_size: (r_c_idx + 1) * rule_chunk_size])

        return external_chunks

    @classmethod
    def generate_evr_eval_instances(cls, instances):
        for instance in instances:
            all_chunks = cls.get_evr_chunks(instance)

            instance["external_chunks"] = {
                "chunk_" + str(chunk_idx): {
                    "statement_" + str(statement_idx): statement
                    for statement_idx, statement in enumerate(chunk)
                } for chunk_idx, chunk in enumerate(all_chunks)
            }

            instance["episodic_buffer_dict"] = {
                "episodic_buffer_0": "This is a cartesian tree search task.",
                "episodic_buffer_1": "Chunk 0 can be used to infer the number of items each person had.",
                "episodic_buffer_2": instance["question_string"]
            }

        return instances


if __name__ == "__main__":
    pass
