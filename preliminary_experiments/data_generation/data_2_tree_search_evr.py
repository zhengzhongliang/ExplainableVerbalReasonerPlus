import json
import random
import re
import math
import os

from preliminary_experiments.data_generation.data_2_tree_search import GenerateTreeSearchData
from preliminary_experiments.data_generation.data_base_class import DataBase
from preliminary_experiments.data_generation.dataset_utils import DataUtils


class GenerateEVRTreeSearchData(DataBase):

    @classmethod
    def split_to_chunks(cls, statements, rules, s_chunk_size=5, r_chunk_size=3):

        num_s_chunks = math.ceil(len(statements)/s_chunk_size)
        statement_chunks = [statements[idx * s_chunk_size: (idx + 1) * s_chunk_size] for idx in range(num_s_chunks)]

        num_r_chunks = math.ceil(len(rules)/r_chunk_size)
        rule_chunks = [rules[idx * r_chunk_size: (idx + 1) * r_chunk_size] for idx in range(num_r_chunks)]

        return statement_chunks, rule_chunks

    @classmethod
    def statement_formal_rep_to_nl(cls, formal_rep):

        main_char = formal_rep[0]
        num_items = formal_rep[1]
        items = formal_rep[2]

        return (f"{main_char} had {num_items} {items[1]}" if num_items != 1 else
                f"{main_char} had {num_items} {items[0]}")

    @classmethod
    def statement_formal_rep_to_nl_question(cls, formal_rep):

        main_char = formal_rep[0]
        num_items = formal_rep[1]
        items = formal_rep[2]

        return (f"Did {main_char} have {num_items} {items[1]}?" if num_items != 1 else
                f"Did {main_char} have {num_items} {items[0]}?")

    @classmethod
    def generate_pattern_gen_prog_1_data(cls, query, instance, search_depth):

        query_nl = cls.statement_formal_rep_to_nl(query)
        query_question_nl = cls.statement_formal_rep_to_nl_question(query)

        episodic_buffers = [
            "This is a tree search task.",
            query_question_nl,
        ]

        target_list = [
            f"#0 = 'Which chunk can prove {query_nl}?'; ",
            "new_mem(#0); "
        ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_2_data(cls,
                                         query,
                                         instance,
                                         search_depth,
                                         with_result,
                                         num_total_chunks,
                                         proof_chk_idx
                                         ):
        """Generate the pattern 2 data for program generation.

        The pattern 2 data needs to be generated at two different places:
         - At the beginning of each recursive search.
         - After each chunk's result is returned.

        :param query: the query to handle
        :param instance: one tree-search instance
        :param search_depth: the current depth of the recursive search.
        :param with_result: whether this pattern has the results.
        :param num_total_chunks: total number of chunks
        :param proof_chk_idx: the index of the chunk which can successfully prove the query.
        Returns:
             - evr_instance: one evr instance for tree-search.
        """

        query_nl = cls.statement_formal_rep_to_nl(query)

        episodic_buffers = [f"Which chunk can prove {query_nl}?"]

        if not with_result:
            target_list = [
                "while check_next_chunk(); ",
                "#0 = get_next_chunk_num(); ",
                f"#1 = 'Can this chunk prove {query_nl}?'; ",
                "#2 = rewrite(#1, #0); ",
                "new_mem(#2); ",
                "end_while; ",
            ]
        else:
            # pattern2_chunk_result_buffer: a list of results of the chunks, about whether each chunk can prove
            #                 the query statement, in natural language expression.

            pattern2_chunk_result_buffer = []
            for chk_idx_ in range(num_total_chunks):
                pattern2_chunk_result_buffer.append(f"Chunk {chk_idx_} can not prove {query_nl}.")

            if proof_chk_idx != -1:
                pattern2_chunk_result_buffer[proof_chk_idx] = f"Chunk {proof_chk_idx} can prove {query_nl}."

            episodic_buffers.extend(pattern2_chunk_result_buffer)

            target_list = [
                "clear_mem(); ",
            ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 2,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_3_data(cls, query, instance, search_depth, chk_idx):

        query_nl = cls.statement_formal_rep_to_nl(query)

        episodic_buffers = [
            f"Can chunk {chk_idx} be used to prove {query_nl}?"
        ]

        target_list = [
            f"#0 = get_chunk('chunk_{chk_idx}'); ",
            "#1 = qa(#0, episodic_buffer_0); ",
            "add_to_episodic(#1); ",
        ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 3,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_4_data(cls,
                                         query,
                                         instance,
                                         search_depth,
                                         chk_idx,
                                         is_statement_chunk,
                                         statement_chunk_result,
                                         matched_rules_one_chunk,
                                         matched_rules_result_list
                                         ):
        """
        Generate the pattern 4 data depending on different scenarios.

        All possible conditions are:
         - statement chunk
            - this statement chunk can prove the query.
            - this statement chunk can not prove the query.
         - rule chunk
            - this rule chunk can not prove the query.
            - this rule chunk can potentially be used to prove the query.
            - this rule chunk can potentially be used to prove the query, and the results of the preconditions are
                returned.

        :param query: the query to be proved
        :param instance: the original tree-search instance
        :param search_depth: depth of search
        :param chk_idx: the chunk index
        :param is_statement_chunk: whether we have handling a statement chunk (not rule chunk).
        :param statement_chunk_result: whether the statement chunk can prove the query (a True or False variable).
        :param matched_rules_one_chunk: a list of matched rules in this chunk (in structured representation).
        :param matched_rules_result_list: whether the matched rule's preconditions can be proved (a list of True/False
            variables). The length of rule_chunk_result_list should be the same as matched_rules_one_chunk.
        :return:
        """

        query_nl = cls.statement_formal_rep_to_nl(query)

        episodic_buffers = [
            f"Can chunk {chk_idx} be used to prove {query_nl}?",
        ]

        # Handles pattern 4.1: fact chunk
        if is_statement_chunk:
            if statement_chunk_result:
                episodic_buffers.append(
                    f"Chunk {chk_idx} can prove {query_nl}."
                )
            else:
                episodic_buffers.append(
                    f"Chunk {chk_idx} can not prove {query_nl}."
                )

            target_list = [
                "return(episodic_buffer_1); ",
            ]

        # Handles pattern 4.2: rule chunk
        else:
            # First generate the goal, could be:
            #   4.2.1: I need to prove A
            #   4.2.2: I need to prove A or B
            #   4.2.3: I need to prove A and B
            #   4.2.4: I need to prove A and B, or C
            #   4.2.5: I need to prove A, or B and C
            #   4.2.6: No chunks can prove ...

            # Handles pattern 4.2.6
            if len(matched_rules_one_chunk) == 0:
                episodic_buffers.append(
                    f"Chunk {chk_idx} can not prove {query_nl}."
                )

                target_list = [
                    "return(episodic_buffer_1); ",
                ]
            # Handles pattern 4.2.1, 4.2.2, 4.2.3, 4.2.4, 4.2.5
            else:
                literals_to_prove = []
                for r in matched_rules_one_chunk:
                    literals_to_prove.append([cls.statement_formal_rep_to_nl(precond)
                                              for precond in r[0]])

                literals_to_prove = [" and ".join(l) if len(l) > 1 else l[0] for l in literals_to_prove]

                literals_to_prove_str = ", or ".join(literals_to_prove)

                pattern4_goal_literal = f"I need to prove {literals_to_prove_str}."

                episodic_buffers.append(pattern4_goal_literal)

                if len(matched_rules_result_list) == 0:
                    target_list = [
                        "#0 = subqs(); ",
                        "for #1 in #0; ",
                        "new_mem(#1); ",
                        "end_for; "
                    ]

                else:
                    # Loop over the literals to prove and add to episodic buffer if the result buffer is not empty.
                    # The format of the result list:
                    #   [statement] is True or [statement] is False
                    for lit_idx, literal_check_result in enumerate(matched_rules_result_list):
                        if literal_check_result:
                            episodic_buffers.append(f"{literals_to_prove[lit_idx]} is True.")
                        else:
                            episodic_buffers.append(f"{literals_to_prove[lit_idx]} is not True.")

                    literal_proved = True if True in matched_rules_result_list else False

                    query_nl = cls.statement_formal_rep_to_nl(query)

                    if literal_proved:
                        target_list = [
                            f"#0 = 'Chunk {chk_idx} can prove {query_nl}.'; ",
                            "return(#0); "
                        ]
                    else:
                        target_list = [
                            f"#0 = 'Chunk {chk_idx} can not prove {query_nl}.'; ",
                            "return(#0); "
                        ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 4,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_5_data(cls,
                                         instance,
                                         search_depth,
                                         statements_to_prove,
                                         statements_proof_results,
                                         ):
        """
        Generate the pattern 5 data, according to different conditions.

        There are four possibly types of input and output:
         - type 1:
            - input: I want to prove [statement 1]
            - output: this is a tree search task; [statement 1 as question]
         - type 2:
            - input: I want to prove [statement 1]; chunk k can prove/no chunks can prove.
            - output: return xxx.
         - type 3:
            - input: I want to prove [statement 1] and [statement 2]
            - output: this is a tree search task; [statement 1 as question]; .....
         - type 4:
            - input: I want to prove [statement 1] and [statement 2]
            - output: return xxx.

        :param instance: an instance for the original tree-search data.
        :param search_depth: the depth of data to deal with
        :param statements_to_prove: the statements to prove
        :param statements_proof_results: the results of the statements to prove, about whether each one can be proved.
            It is a list of chunk numbers. It will have -1 if no chunks can prove the statement.
        :return: evr_instance: an evr instance for this pattern.
        """

        statements_nl = [cls.statement_formal_rep_to_nl(s) for s in statements_to_prove]

        if len(statements_to_prove) == 1:
            if len(statements_proof_results) == 0:
                episodic_buffers = [f"I need to prove {statements_nl[0]}."]

                target_list = [
                    "#0 = 'This is a tree search task.'; ",
                    "#1 = subqs(); ",
                    "for #2 in #1; ",
                    "new_mem(#0, #2); ",
                    "end_for; "
                ]
            else:
                episodic_buffers = [f"I need to prove {statements_nl[0]}."]

                statement_result = (f"Chunk {statements_proof_results[0]} can prove {statements_nl[0]}."
                                    if statements_proof_results[0] != -1 else
                                    f"No chunks can prove {statements_nl[0]}.")

                episodic_buffers.append(statement_result)

                target_list = [
                    (f"#0 = '{statements_nl[0]} is True.'; "
                     if statements_proof_results[0] != -1 else
                     f"#0 = '{statements_nl[0]} is not True.'; "),
                    "return(#0); "
                ]

        else:
            statements_nl_comb = " and ".join(statements_nl)

            if len(statements_proof_results) == 0:
                episodic_buffers = [f"I need to prove {statements_nl_comb}."]

                target_list = [
                    "#0 = 'This is a tree search task.'; ",
                    "#1 = subqs(); ",
                    "for #2 in #1; ",
                    "new_mem(#0, #2); ",
                    "end_for; "
                ]

            else:
                episodic_buffers = [f"I need to prove {statements_nl_comb}."]

                statements_results = [
                    f"Chunk {s_p_r} can prove {statements_nl[p_r_idx]}."
                    if s_p_r != -1 else
                    f"No chunks can prove {statements_nl[p_r_idx]}."
                    for p_r_idx, s_p_r in enumerate(statements_proof_results)
                ]

                episodic_buffers.extend(statements_results)

                target_list = [
                    (f"#0 = '{statements_nl_comb} is True.'; "
                     if -1 not in statements_proof_results else
                     f"#0 = '{statements_nl_comb} is not True.'; "),
                    "return(#0); "
                ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 5,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_6_data(cls,
                                         query,
                                         instance,
                                         search_depth,
                                         proof_chk_idx):
        """
        Generate pattern 6 data for evr training.

        :param query: the query to handle
        :param instance: one tree-search instance
        :param search_depth: the current depth of search
        :param proof_chk_idx: the chunk idx that can successfully prove the query.
        :return:
        """
        query_nl = cls.statement_formal_rep_to_nl(query)

        if proof_chk_idx == -1:
            chunk_proof_result_nl = f"No chunks can prove {query_nl}."
        else:
            chunk_proof_result_nl = f"Chunk {proof_chk_idx} can prove {query_nl}."

        episodic_buffers = [f"Which chunk can prove {query_nl}?", chunk_proof_result_nl]

        target_list = [
            "return(episodic_buffer_1); ",
        ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 6,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_gen_prog_7_data(cls,
                                         query,
                                         instance,
                                         search_depth,
                                         proof_chk_idx):
        """
        Generate pattern 7 data, which returns the final proof result of a statement.

        :param query: the current query to be proved.
        :param instance: the original tree search instance.
        :param search_depth: search depth
        :param proof_chk_idx: the chunk idx that can prove the query. -1 means no chunks can prove it.
        :return:
        """

        query_nl = cls.statement_formal_rep_to_nl(query)
        query_question_nl = cls.statement_formal_rep_to_nl_question(query)

        # statement_proof_result: a natural language statement, one of the following two options:
        #                  - Chunk k can prove [statement].
        #                  - No chunks can prove [statement].
        statement_proof_result = (
            f"Chunk {proof_chk_idx} can prove {query_nl}." if proof_chk_idx != -1 else
            f"No chunks can prove {query_nl}."
        )

        episodic_buffers = [
            "This is a tree search task.",
            f"{query_question_nl}",
            f"{statement_proof_result}"
        ]

        target_list = [
            "return(episodic_buffer_2); ",
        ]

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "generate_program: " + context
        target = "".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 7,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_rewrite_1_data(cls,
                                        query,
                                        instance,
                                        search_depth,
                                        chk_idx):

        query_nl = cls.statement_formal_rep_to_nl(query)

        input_text_list = [
            f"Can this chunk prove {query_nl}?",
            f"chunk_{chk_idx}"
        ]

        context = " ".join(input_text_list)
        input_text = "rewrite: " + context
        target = f"Can chunk {chk_idx} be used to prove {query_nl}?"

        evr_instance = {
            "task": "rewrite",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_qa_1_data(cls,
                                   query,
                                   instance,
                                   search_depth,
                                   chk_idx,
                                   chunk_sents_list,
                                   is_statement_chunk,
                                   statement_chunk_result,
                                   matched_rules):

        query_nl = cls.statement_formal_rep_to_nl(query)

        chunk_text = " ".join([f"statement_{i_}: {s_}"
                               for i_, s_ in enumerate(chunk_sents_list)])

        input_text_list = [
            chunk_text,
            f"Can chunk {chk_idx} be used to prove {query_nl}?"
        ]
        if is_statement_chunk:
            if statement_chunk_result:
                target = f"Chunk {chk_idx} can prove {query_nl}."
            else:
                target = f"Chunk {chk_idx} can not prove {query_nl}."
        else:
            if len(matched_rules) > 0:
                disj_literals_nl = []
                for r_ in matched_rules:
                    matched_rule_preconds = r_[0]
                    conj_literal = " and ".join([cls.statement_formal_rep_to_nl(p) for p in matched_rule_preconds])
                    disj_literals_nl.append(conj_literal)
                disj_literals_nl = ", or ".join(disj_literals_nl)
                target = f"I need to prove {disj_literals_nl}."
            else:
                target = f"Chunk {chk_idx} can not prove {query_nl}."

        context = " ".join(input_text_list)
        input_text = "qa: " + context

        evr_instance = {
            "task": "qa",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_subqs_1_data(cls,
                                      query,
                                      instance,
                                      search_depth,
                                      chk_idx,
                                      matched_rules):

        query_nl = cls.statement_formal_rep_to_nl(query)

        disj_literals_nl = []
        for r_ in matched_rules:
            matched_rule_preconds = r_[0]
            conj_literal = " and ".join([cls.statement_formal_rep_to_nl(p) for p in matched_rule_preconds])
            disj_literals_nl.append(conj_literal)
        disj_literals_nl = ", or ".join(disj_literals_nl)

        input_text_list = [
            f"Can chunk {chk_idx} be used to prove {query_nl}?",
            f"I need to prove {disj_literals_nl}."
        ]

        if len(matched_rules) == 1:

            # First handle the situation when there is only one literal in the precondition
            target = f"['I need to prove {disj_literals_nl}.']"

        else:
            new_literals_ = disj_literals_nl.split(", or ")
            new_literals = [f"'I need to prove {lit}.'" for lit in new_literals_]
            new_literals = ", ".join(new_literals)
            target = f"[{new_literals}]"

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(input_text_list)])
        input_text = "subqs: " + context

        evr_instance = {
            "task": "subqs",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_subqs_2_data(cls,
                                      query,
                                      instance,
                                      search_depth,
                                      preconds):

        preconds_nl = " and ".join([cls.statement_formal_rep_to_nl(pc) for pc in preconds])

        input_text_list = [
            f"I need to prove {preconds_nl}.",
        ]

        target_text_list = [cls.statement_formal_rep_to_nl_question(p_c) for p_c in preconds]
        target_text_list = [f"'{t}'" for t in target_text_list]
        target = ", ".join(target_text_list)
        target = f"[{target}]"

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(input_text_list)])
        input_text = "subqs: " + context

        evr_instance = {
            "task": "subqs",
            "pattern": 2,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def generate_pattern_clear_mem_1_data(cls,
                                          query,
                                          instance,
                                          search_depth,
                                          num_all_chunks,
                                          proof_chk_idx
                                          ):
        """
        Generate the clear_mem data based on pattern 2.

        :param query:
        :param instance:
        :param search_depth:
        :param num_all_chunks: the number of all chunks
        :param proof_chk_idx: the index of the chunk that can successfully prove the query.
        :return:
        """
        all_chunks_results = [False] * num_all_chunks
        if proof_chk_idx >= 0:
            all_chunks_results[proof_chk_idx] = True

        query_nl = cls.statement_formal_rep_to_nl(query)

        episodic_buffers = [
            f"Which chunk can prove {query_nl}?"
        ]

        target_episodic_buffers = [
            f"Which chunk can prove {query_nl}?"
        ]

        for chk_idx, chunk_result in enumerate(all_chunks_results):
            if chunk_result:
                episodic_buffers.append(f"Chunk {chk_idx} can prove {query_nl}.")
            else:
                episodic_buffers.append(f"Chunk {chk_idx} can not prove {query_nl}.")

        if True in all_chunks_results:
            answer_chk_idx = all_chunks_results.index(True)
            target_episodic_buffers.append(f"Chunk {answer_chk_idx} can prove {query_nl}.")
        else:
            target_episodic_buffers.append(f"No chunks can prove {query_nl}.")

        context = " ".join([f"episodic_buffer_{e_idx}: {ep}" for e_idx, ep in enumerate(episodic_buffers)])
        input_text = "clear_mem: " + context
        target = " ".join([f"episodic_buffer_{e_idx}: '{ep}'" for e_idx, ep in enumerate(target_episodic_buffers)])

        evr_instance = {
            "task": "clear_mem",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "search_depth": search_depth
        }

        return evr_instance

    @classmethod
    def evr_instance_backward_recursive(
            cls, query, instance,
            all_statements_by_chunk, all_rules_by_chunk, all_statements_nl_by_chunk, all_rules_nl_by_chunk,
            traversal_history, depth_history, evr_instances,
            d=0, debug_flag=False):
        """
        Recursively generates the evr tree search instance.

        In each recursive level, the following functions are called:
         - pattern_1 function: called one time
         - pattern_2 function: called two times, one in the beginning, one after the chunks results are collected.
         - pattern_3 function: called one time, in the for loop to judge the chunks (one time each chunk).
         - pattern_4 function: called in each for loop to check the conditions.
            The difference between pattern 3 and pattern 4: pattern 3 doesn't have the returned results, pattern 4 has.
         - pattern_5 function: only called in each loop to check the rules, and only called when the rule can be used
            to prove the query.
         - pattern_6 function: only called once, after all chunks results are collected.
         - pattern_7 function: only called once, after all chunks results are collected.

        Some important variables and their meanings
         - chunk_result: each single chunk's result (True/False)

        :param query: the query to prove
        :param instance: the original tree-search instance
        :param all_statements_by_chunk: all statements split to chunks
        :param all_rules_by_chunk: all rules split to chunks
        :param all_statements_nl_by_chunk: all statements split to chunks, but in natural language format
        :param all_rules_nl_by_chunk: all rules split to chunks, but in natural language format
        :param traversal_history:
        :param depth_history:
        :param evr_instances:
        :param d: search depth
        :param debug_flag:
        :return:
        """

        query_nl = cls.statement_formal_rep_to_nl(query)
        all_chunks = all_statements_by_chunk + all_rules_by_chunk
        all_chunks_nl = all_statements_nl_by_chunk + all_rules_nl_by_chunk

        # Generate pattern 1 data
        evr_instances.append(cls.generate_pattern_gen_prog_1_data(
            query=query, instance=instance, search_depth=d))

        # Generate pattern 2 data
        evr_instances.append(cls.generate_pattern_gen_prog_2_data(
            query=query, instance=instance, search_depth=d, with_result=False, num_total_chunks=len(all_chunks),
            proof_chk_idx=-1))

        # Query proved flag: whether this query can be proved in this recursive level. This could be either the
        # statements chunk, or rule chunk.
        query_proved_flag = False

        # Stores which chunk can be used to prove the query.
        proof_chk_idx = -1

        for chk_idx, s_or_r_chunk in enumerate(all_chunks):

            # Generate rewrite pattern 1 data
            evr_instances.append(cls.generate_pattern_rewrite_1_data(
                query=query, instance=instance, search_depth=d, chk_idx=chk_idx
            ))

            # Generate pattern 3 data
            evr_instances.append(cls.generate_pattern_gen_prog_3_data(query=query,
                                                                      instance=instance,
                                                                      search_depth=d,
                                                                      chk_idx=chk_idx))

            # Handle statement chunk.
            if chk_idx < len(all_statements_by_chunk):
                statements_one_chunk = s_or_r_chunk

                if query in statements_one_chunk:

                    query_proved_flag = True
                    proof_chk_idx = chk_idx
                    chunk_result = True

                    # Keep track of the traversal and depth history for debugging purpose.
                    traversal_history.append((chk_idx, statements_one_chunk.index(query)))
                    depth_history.append(d)

                else:
                    chunk_result = False

                # Generate the pattern qa 1 data.
                evr_instances.append(cls.generate_pattern_qa_1_data(
                    query=query, instance=instance, search_depth=d, chk_idx=chk_idx,
                    chunk_sents_list=all_chunks_nl[chk_idx], is_statement_chunk=True,
                    statement_chunk_result=chunk_result,matched_rules=[]
                ))

                # Generate the pattern 4 data.
                evr_instances.append(cls.generate_pattern_gen_prog_4_data(
                    query=query, instance=instance, search_depth=d, chk_idx=chk_idx,
                    is_statement_chunk=True, statement_chunk_result=chunk_result,
                    matched_rules_one_chunk=[], matched_rules_result_list=[]))

            # Handle rule chunk.
            else:
                rules_one_chunk = s_or_r_chunk

                matched_rules_list = []
                matched_rules_proof_result_list = []

                for r_idx_, rule in enumerate(rules_one_chunk):

                    rule_match_flag = rule[1] == query

                    if rule_match_flag:
                        matched_rules_list.append(rule)

                        # Keep track of the traversal and depth history for debugging purposes.
                        r_idx = rules_one_chunk.index(rule)
                        traversal_history.append((chk_idx, r_idx))
                        depth_history.append(d)

                        all_precond_sat = []
                        all_precond_proof_chunk_indices = []

                        # Pattern 5 data needs to be generated two times:
                        # In the first generation, no results are included.
                        # In the second generation, the results are taken into consideration.
                        evr_instances.append(cls.generate_pattern_gen_prog_5_data(
                            instance=instance, search_depth=d,
                            statements_to_prove=rule[0],
                            statements_proof_results=[],
                        ))

                        for precond in rule[0]:
                            s_proved_flag, s_proof_chk_idx, traversal_history, depth_history, evr_instances = \
                                cls.evr_instance_backward_recursive(
                                    precond, instance,
                                    all_statements_by_chunk, all_rules_by_chunk,
                                    all_statements_nl_by_chunk, all_rules_nl_by_chunk,
                                    traversal_history, depth_history, evr_instances,
                                    d=d + 1, debug_flag=debug_flag
                                )

                            all_precond_sat.append(s_proved_flag)
                            all_precond_proof_chunk_indices.append(s_proof_chk_idx)

                        evr_instances.append(cls.generate_pattern_gen_prog_5_data(
                            instance=instance, search_depth=d,
                            statements_to_prove=rule[0],
                            statements_proof_results=all_precond_proof_chunk_indices,
                        ))

                        evr_instances.append(cls.generate_pattern_subqs_2_data(
                            query=query, instance=instance, search_depth=d, preconds=rule[0]
                        ))

                        if False not in all_precond_sat:
                            query_proved_flag = True
                            proof_chk_idx = chk_idx
                            matched_rules_proof_result_list.append(True)
                        else:
                            matched_rules_proof_result_list.append(False)

                if len(matched_rules_list) > 0:
                    # Generate the pattern subqs 1 data.
                    evr_instances.append(cls.generate_pattern_subqs_1_data(
                        query=query, instance=instance, search_depth=d, chk_idx=chk_idx,
                        matched_rules=matched_rules_list
                    ))

                # Generate the pattern qa 1 data.
                evr_instances.append(cls.generate_pattern_qa_1_data(
                    query=query, instance=instance, search_depth=d, chk_idx=chk_idx,
                    chunk_sents_list=all_chunks_nl[chk_idx], is_statement_chunk=False,
                    statement_chunk_result=None, matched_rules=matched_rules_list
                ))

                # Generate the first pattern 4 data. Depending on the number of matched rules, it could be one of the
                # following situation:
                #    - Chunk k can not prove ....
                #    - I need to prove XX, or YY.
                evr_instances.append(cls.generate_pattern_gen_prog_4_data(
                    query=query, instance=instance, search_depth=d, chk_idx=chk_idx,
                    is_statement_chunk=False, statement_chunk_result=None,
                    matched_rules_one_chunk=matched_rules_list,
                    matched_rules_result_list=[]
                ))

                # If there is anything that we need to further prove, generate one more pattern 4 data with the
                # proof results of the matched literals.
                # E.g., could be something like "XX is True. YY is False."
                if len(matched_rules_list) > 0:
                    evr_instances.append(cls.generate_pattern_gen_prog_4_data(
                        query=query, instance=instance, search_depth=d, chk_idx=chk_idx,
                        is_statement_chunk=False, statement_chunk_result=None,
                        matched_rules_one_chunk=matched_rules_list,
                        matched_rules_result_list=matched_rules_proof_result_list
                    ))

        # Generate pattern 2 data
        evr_instances.append(cls.generate_pattern_gen_prog_2_data(
            query=query, instance=instance, search_depth=d,
            with_result=True, num_total_chunks=len(all_chunks), proof_chk_idx=proof_chk_idx
        ))

        # Generate pattern 6 data
        evr_instances.append(cls.generate_pattern_gen_prog_6_data(
            query=query, instance=instance, search_depth=d,
            proof_chk_idx=proof_chk_idx
        ))

        # Generate pattern 7 data
        evr_instances.append(cls.generate_pattern_gen_prog_7_data(
            query=query, instance=instance, search_depth=d,
            proof_chk_idx=proof_chk_idx
        ))

        # Generate pattern clear mem data
        evr_instances.append(cls.generate_pattern_clear_mem_1_data(
            query=query, instance=instance, search_depth=d, num_all_chunks=len(all_chunks), proof_chk_idx=proof_chk_idx
        ))

        return query_proved_flag, proof_chk_idx, traversal_history, depth_history, evr_instances

    @classmethod
    def generate_evr_data_one_instance(cls, instance):

        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]
        # Only the initial grounded statement should be added

        all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in instance["rules"]["backtracking"] if r is not None]

        # This way the order of the formal representations can be associated with the natural language representations
        all_statements = [all_statements[int(old_idx)]
                          for old_idx in instance["statement_indices_shuffle_map"].keys()]

        all_rules = [all_rules[int(old_idx)]
                     for old_idx in instance["rule_indices_shuffle_map"].keys()]

        all_statements_nl = [GenerateTreeSearchData.convert_formal_statement_to_natural_language(s)
                             for s in all_statements]

        all_rules_nl = [GenerateTreeSearchData.convert_formal_rules_to_natural_language(r)
                        for r in all_rules]

        statement_chunks, rule_chunks = cls.split_to_chunks(all_statements, all_rules)
        statement_nl_chunks, rule_nl_chunks = cls.split_to_chunks(all_statements_nl, all_rules_nl)

        query = instance["question"]

        (query_proved_flag, proof_chk_idx, traversal_history, depth_history,
         evr_instances) = cls.evr_instance_backward_recursive(
            query, instance,
            statement_chunks, rule_chunks,
            statement_nl_chunks, rule_nl_chunks,
            traversal_history=[], depth_history=[], evr_instances=[],
            d=0, debug_flag=False
        )

        return (evr_instances, query_proved_flag, proof_chk_idx,
                statement_nl_chunks, rule_nl_chunks, traversal_history, depth_history)

    @classmethod
    def generate_evr_instances(cls, instances):

        evr_instances_all = []
        for instance in instances:

            (evr_instances, query_proved_flag, proof_chk_idx, statement_nl_chunks, rule_nl_chunks,
             traversal_history, depth_history) = cls.generate_evr_data_one_instance(instance)

            evr_instances_all.extend(evr_instances)

        return evr_instances_all

    @classmethod
    def get_evr_chunks(cls, instance):

        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]

        all_statements_nl = instance["context_list"][:len(all_statements)]

        all_rules_nl = instance["context_list"][len(all_statements):]

        statement_nl_chunks, rule_nl_chunks = cls.split_to_chunks(all_statements_nl, all_rules_nl)

        return statement_nl_chunks + rule_nl_chunks

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
                "episodic_buffer_0": "This is a tree search task.",
                "episodic_buffer_1": instance["question_string"]
            }

        return instances


if __name__ == "__main__":
    pass
    #GenerateEVRTreeSearchData.debug_check_instance_structure()
    #GenerateEVRTreeSearchData.debug_check_splitting_chunks()
