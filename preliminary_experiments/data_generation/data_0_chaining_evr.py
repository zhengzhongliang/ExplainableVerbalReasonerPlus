import json

from preliminary_experiments.data_generation.data_utils import DataUtils


class GenerateEVRChainingData:

    """
    Generate each pattern of data for EVR. Note that each pattern should have two sub patterns: one for
    controller one for executor.
    """

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
    def generate_pattern_qa_1_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        initial_statements = [chain["context_list"][0] for chain in instance["chains"]]
        context = " ".join(
            ["statement_" + str(s_idx) + ": " + s for s_idx, s in enumerate(initial_statements)])
        question = "According to chunk 0, how many " + item_tuple[
            1] + " did " + main_chara + " have in the beginning?"

        input_text = f"qa: {context} {question}"

        target = "According to chunk 0, " + main_chara + " had " + \
                 str(selected_chain["formal_reps"][0][1]) + " " + \
                 cls.get_item_by_quantity(selected_chain["formal_reps"][0][1], item_tuple) + " in the beginning."

        evr_instance = {
            "task": "qa",
            "pattern": 1,
            "context": context,
            "question": question,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_qa_2_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        if instance["depth"] != 0:
            for c_idx, chain in enumerate(instance["chains"]):

                main_chara = selected_chain["formal_reps"][0][0]
                item_tuple = selected_chain["formal_reps"][0][2]

                context = " ".join(["statement_" + str(s_idx) + ": " + s for s_idx, s in enumerate(chain["context_list"][1:])])

                question = (f"Can this chunk be used to infer how many {item_tuple[1]} {main_chara} "
                            f"had after exchanging?")

                input_text = f"qa: {context} {question}"

                target = "True" if c_idx == selected_chain_idx else "False"

                evr_instance = {
                    "task": "qa",
                    "pattern": 2,
                    "context": context,
                    "question": question,
                    "input": input_text,
                    "target": target,
                    "org_id": instance["id"],
                    "depth": instance["depth"],
                    "selected_chain_idx": selected_chain_idx
                }

                evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_qa_3_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        if instance["depth"] != 0:

            main_chara = selected_chain["formal_reps"][0][0]
            item_tuple = selected_chain["formal_reps"][0][2]

            beginning_quantity = selected_chain["formal_reps"][0][1]
            up_to_date_quantity = beginning_quantity
            for ctx_idx, ctx in enumerate(selected_chain["context_list"][1:]):
                item_nl_ = cls.get_item_by_quantity(up_to_date_quantity, item_tuple)
                context = (f"{main_chara} had {up_to_date_quantity} {item_nl_}. "
                           f"{ctx}")
                question = f"How many {item_tuple[1]} did {main_chara} have after exchanging?"

                input_text = f"qa: {context} {question}"

                delta_quantity = selected_chain["formal_reps"][ctx_idx + 1][1]
                up_to_date_quantity += delta_quantity
                target = f"{main_chara} had {up_to_date_quantity} {item_tuple[1]}."

                evr_instance = {
                    "task": "qa",
                    "pattern": 3,
                    "context": context,
                    "question": question,
                    "input": input_text,
                    "target": target,
                    "org_id": instance["id"],
                    "depth": instance["depth"],
                    "selected_chain_idx": selected_chain_idx
                }

                evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_qa_4_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []
        if instance["depth"] == 0:

            main_chara = selected_chain["formal_reps"][0][0]
            item_tuple = selected_chain["formal_reps"][0][2]

            beginning_quantity = selected_chain["formal_reps"][0][1]
            item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

            context_list = [
                f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning.",
                f"No one exchanged items with others."
            ]
            context = " ".join(context_list)
            question = f"How many {item_tuple[1]} did {main_chara} have in the end?"
            input_text = f"qa: {context} {question}"

            target = str(instance["answer"])

            evr_instance = {
                "task": "qa",
                "pattern": 4,
                "context": context,
                "question": question,
                "input": input_text,
                "target": target,
                "org_id": instance["id"],
                "depth": instance["depth"],
                "selected_chain_idx": selected_chain_idx
            }

            evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_1_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        num_chains = len(instance["chains"])

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        question = (f"According to the chunks from chunk {1} to chunk {num_chains}, "
                    f"which chunk can be used to infer how many {item_tuple[1]} {main_chara} had after exchanging?")

        response = str(selected_chain_idx + 1)

        input_text = f"rewrite: {question} {response}"

        target = (f"Chunk {selected_chain_idx + 1} can be used to infer "
                  f"how many {item_tuple[1]} {main_chara} had after exchanging.")

        evr_instance = {
            "task": "rewrite",
            "pattern": 1,
            "question": question,
            "response": response,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_2_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        answer_quantity = int(instance["answer"])
        item_nl = cls.get_item_by_quantity(answer_quantity, item_tuple)

        question = (f"According to chunk {selected_chain_idx + 1}, "
                    f"how many {item_tuple[1]} did {main_chara} have after exchanging?")
        response = f"{main_chara} had {answer_quantity} {item_nl}."

        input_text = f"rewrite: {question} {response}"

        target = f"{main_chara} had {answer_quantity} {item_nl} after exchanging."

        evr_instance = {
            "task": "rewrite",
            "pattern": 2,
            "question": question,
            "response": response,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_3_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        item_nl = cls.get_item_by_quantity(int(instance["answer"]), item_tuple)

        question = f"How many {item_tuple[1]} did {main_chara} have in the end?"

        response = f"{main_chara} had {instance['answer']} {item_nl} after exchanging."

        input_text = f"rewrite: {question} {response}"

        target = str(instance["answer"])

        evr_instance = {
            "task": "rewrite",
            "pattern": 3,
            "question": question,
            "response": response,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_clear_mem_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        n_chains = len(instance["chains"])

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            "This is a chaining task.",
            "Chunk 0 answers how many items each person had in the beginning.",
            f"Chunk 1 to chunk {n_chains} can be used to infer how many items each person had after exchanging.",
            f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?",
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning."
        ]

        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])

        input_text = "clear_mem: " + context

        target_list = [
            "'This is a chaining task.'",
            f"'Chunk 1 to chunk {n_chains} can be used to infer how many items each person had after exchanging.'",
            f"'{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?'",
            f"'According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning.'"
        ]
        target = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(target_list)
        ])

        evr_instance = {
            "task": "clear_mem",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_1_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        n_chains = len(instance["chains"])

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        beginning_quantity = selected_chain["formal_reps"][0][1]

        if instance["depth"] == 0:
            context_list = [
                "This is a chaining task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?"
            ]
        else:
            context_list = [
                "This is a chaining task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk 1 to chunk {n_chains} can be used to infer how many items each person had after exchanging.",
                f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?"
            ]
        context = " ".join([
           f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])

        input_text = "generate_program: " + context

        target_list = [
            f"#0 = 'According to chunk 0, how many {item_tuple[1]} did {main_chara} have in the beginning?';",
            "new_mem(#0);"
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 1,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_2_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        beginning_quantity = selected_chain["formal_reps"][0][1]

        context_list = [
            f"According to chunk 0, how many {item_tuple[1]} did {main_chara} have in the beginning?"
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])

        input_text = "generate_program: " + context

        target_list = [
            "#0 = 'chunk_0';",
            "#1 = get_chunk(#0);",
            "#2 = qa(#1, episodic_buffer_0);",
            "add_to_episodic(#2);",
        ]
        target = " ".join(target_list)

        # The length (number of tokens) of the prediction is usually 136 or 137

        evr_instance = {
            "task": "generate_program",
            "pattern": 2,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_3_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            f"According to chunk 0, how many {item_tuple[1]} did {main_chara} have in the beginning?",
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning."
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])

        input_text = "generate_program: " + context

        target = "return(episodic_buffer_1);"

        evr_instance = {
            "task": "generate_program",
            "pattern": 3,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_4_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        if instance["depth"] == 0:
            context_list = [
                "This is a chaining task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                "No one exchanged items with others.",
                f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?",
                f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning."
            ]
        else:
            context_list = [
                "This is a chaining task.",
                "Chunk 0 answers how many items each person had in the beginning.",
                f"Chunk 1 to chunk {n_chains} can be used to infer how many items each person had after exchanging.",
                f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?",
                f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning."
            ]

        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])

        input_text = "generate_program: " + context

        if instance["depth"] == 0:
            target_list = [
                "#0 = qa(episodic_buffer_4, episodic_buffer_2, episodic_buffer_3);",
                "return(#0);"
            ]
        else:
            target_list = ["clear_mem();"]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 4,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_5_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            "This is a chaining task.",
            f"Chunk 1 to chunk {n_chains} can be used to infer how many items each person had after exchanging.",
            f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?",
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning."
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            (f"#0 = 'According to the chunks from chunk 1 to chunk {n_chains}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?';"),
            "new_mem(episodic_buffer_3, #0);"
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 5,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_6_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning.",
            (f"According to the chunks from chunk 1 to chunk {n_chains}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?"),
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            (f"#0 = 'According to the chunks from chunk 1 to chunk {n_chains}, which chunk can be used to infer "
             f"how many {item_tuple[1]} {main_chara} had after exchanging?';"),
            "new_mem(#0);"
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 6,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_7_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            (f"According to the chunks from chunk 1 to chunk {n_chains}, which chunk can be used to infer "
             f"how many {item_tuple[1]} {main_chara} had after exchanging?")
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            f"#0 = list_chunk_nums('chunk_1', 'chunk_{n_chains}');",
            "for #1 in #0;",
            "#2 = get_chunk(#1);",
            f"#3 = 'Can this chunk be used to infer how many {item_tuple[1]} {main_chara} had after exchanging?';",
            "#4 = qa(#2, #3);",
            "if #4 == 'True';",
            "#5 = #1;",
            "else;",
            "pass;",
            "end_if;",
            "end_for;",
            "#6 = rewrite(episodic_buffer_0, #5);",
            "add_to_episodic(#6);",
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 7,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_8_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            (f"According to the chunks from chunk 1 to chunk {n_chains}, which chunk can be used to infer "
             f"how many {item_tuple[1]} {main_chara} had after exchanging?"),
            (f"Chunk {selected_chain_idx + 1} can be used to infer "
             f"how many {item_tuple[1]} {main_chara} had after exchanging.")
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        selected_chunk_name = "chunk_" + str(selected_chain_idx)

        target = "return(episodic_buffer_1);"

        evr_instance = {
            "task": "generate_program",
            "pattern": 8,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_9_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning.",
            (f"According to the chunks from chunk 1 to chunk {n_chains}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?"),
            (f"Chunk {selected_chain_idx + 1} can be used to infer "
             f"how many {item_tuple[1]} {main_chara} had after exchanging.")
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            (f"#0 = 'According to chunk {selected_chain_idx + 1}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?';"),
            "new_mem(episodic_buffer_0, #0);"
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 9,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_10_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        context_list = [
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl} in the beginning.",
            (f"According to chunk {selected_chain_idx + 1}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?")
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            f"#0 = 'chunk_{selected_chain_idx + 1}';",
            f"#1 = '{main_chara} had {beginning_quantity} {item_nl}.';",
            "while check_next_statement(#0);",
            "#2 = get_next_statement_num(#0);",
            "#3 = get_statement(#0, #2);",
            f"#4 = 'How many {item_tuple[1]} did {main_chara} have after exchanging?';",
            "#1 = qa(#1, #3, #4);",
            "end_while;",
            "add_to_episodic(#1);",
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 10,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_11_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl_beginning = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        answer_quantity = selected_chain["answer"]
        item_nl_answer = cls.get_item_by_quantity(answer_quantity, item_tuple)

        context_list = [
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl_beginning} in the beginning.",
            (f"According to chunk {selected_chain_idx + 1}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?"),
            f"{main_chara} had {answer_quantity} {item_nl_answer}."
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            "#0 = rewrite(episodic_buffer_1, episodic_buffer_2);",
            "return(#0);",
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 11,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_12_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl_beginning = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        answer_quantity = selected_chain["answer"]
        item_nl_answer = cls.get_item_by_quantity(answer_quantity, item_tuple)

        context_list = [
            (f"According to chunk 0, "
             f"{main_chara} had {beginning_quantity} {item_nl_beginning} in the beginning."),
            (f"According to the chunks from chunk 1 to chunk {n_chains}, "
             f"how many {item_tuple[1]} did {main_chara} have after exchanging?"),
            (f"Chunk {selected_chain_idx + 1} can be used to infer "
             f"how many {item_tuple[1]} {main_chara} had after exchanging."),
            f"{main_chara} had {answer_quantity} {item_nl_answer} after exchanging."
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target = "return(episodic_buffer_3);"

        evr_instance = {
            "task": "generate_program",
            "pattern": 12,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_13_data(cls, instance, selected_chain_idx, selected_chain):

        evr_instances = []

        main_chara = selected_chain["formal_reps"][0][0]
        item_tuple = selected_chain["formal_reps"][0][2]
        n_chains = len(instance["chains"])

        beginning_quantity = selected_chain["formal_reps"][0][1]
        item_nl_beginning = cls.get_item_by_quantity(beginning_quantity, item_tuple)

        answer_quantity = selected_chain["answer"]
        item_nl_answer = cls.get_item_by_quantity(answer_quantity, item_tuple)

        context_list = [
            "This is a chaining task.",
            f"Chunk 1 to chunk {n_chains} can be used to infer how many items each person had after exchanging.",
            f"{cls.formal_statement_to_nl_question(selected_chain['formal_reps'][0])} in the end?",
            f"According to chunk 0, {main_chara} had {beginning_quantity} {item_nl_beginning} in the beginning.",
            f"{main_chara} had {answer_quantity} {item_nl_answer} after exchanging."
        ]
        context = " ".join([
            f"episodic_buffer_{epi_idx}: {ep}" for epi_idx, ep in enumerate(context_list)
        ])
        input_text = f"generate_program: {context}"

        target_list = [
            "#0 = rewrite(episodic_buffer_2, episodic_buffer_4);",
            "return(#0);"
        ]
        target = " ".join(target_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 13,
            "context": context,
            "input": input_text,
            "target": target,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "selected_chain_idx": selected_chain_idx
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_evr_data_one_instance(cls, instance):

        selected_chain_idx = instance["selected_chain_idx"]
        selected_chain = instance["chains"][selected_chain_idx]

        pattern_generation_func = {
            "qa": {
                1: cls.generate_pattern_qa_1_data,
                2: cls.generate_pattern_qa_2_data,
                3: cls.generate_pattern_qa_3_data,
                4: cls.generate_pattern_qa_4_data,
            },

            "rewrite": {
                1: cls.generate_pattern_rewrite_1_data,
                2: cls.generate_pattern_rewrite_2_data,
                3: cls.generate_pattern_rewrite_3_data,
            },

            "gen_prog": {
                1: cls.generate_pattern_gen_prog_1_data,
                2: cls.generate_pattern_gen_prog_2_data,
                3: cls.generate_pattern_gen_prog_3_data,
                4: cls.generate_pattern_gen_prog_4_data,
                5: cls.generate_pattern_gen_prog_5_data,
                6: cls.generate_pattern_gen_prog_6_data,
                7: cls.generate_pattern_gen_prog_7_data,
                8: cls.generate_pattern_gen_prog_8_data,
                9: cls.generate_pattern_gen_prog_9_data,
                10: cls.generate_pattern_gen_prog_10_data,
                11: cls.generate_pattern_gen_prog_11_data,
                12: cls.generate_pattern_gen_prog_12_data,
                13: cls.generate_pattern_gen_prog_13_data,
            },

            "clear_mem": {
                1: cls.generate_pattern_clear_mem_data
            }
        }

        evr_instances_all_patterns = []
        for pattern in pattern_generation_func.keys():
            for pattern_num in pattern_generation_func[pattern].keys():
                evr_instances = pattern_generation_func[pattern][pattern_num](
                    instance, selected_chain_idx, selected_chain)

                evr_instances_all_patterns.extend(evr_instances)

        return evr_instances_all_patterns

    @classmethod
    def generate_evr_instances(cls, instances):

        all_evr_instances = []

        for instance in instances:
            all_evr_instances.extend(cls.generate_evr_data_one_instance(instance))

        return all_evr_instances

    @classmethod
    def debug_evr_instances(cls):

        instances = DataUtils.load_json(
            "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/"
            "data_generated/chaining_v0.4/chaining_data_du5.json"
        )["train"]

        import random
        random.seed(0)

        random.shuffle(instances)

        evr_instances_all_patterns = cls.generate_evr_instances(instances)

        for evr_instance in evr_instances_all_patterns:
            print("=" * 40)
            print(json.dumps(evr_instance, indent=2))
            input("-" * 40)

    @classmethod
    def generate_evr_eval_instances(cls, instances):

        for instance in instances:

            selected_chain_idx = instance["selected_chain_idx"]
            selected_chain = instance["chains"][selected_chain_idx]

            main_chara = selected_chain["formal_reps"][0][0]
            item_tuple = selected_chain["formal_reps"][0][2]

            external_chunks = [[chain["context_list"][0] for chain in instance["chains"]]]
            if instance["depth"] >= 1:
                external_chunks.extend([chain["context_list"][1:] for chain in instance["chains"]])

            instance["external_chunks"] = {
                "chunk_" + str(chunk_idx): {
                    "statement_" + str(statement_idx): statement
                    for statement_idx, statement in enumerate(chunk)
                } for chunk_idx, chunk in enumerate(external_chunks)
            }

            num_chains = len(instance["chains"])

            if instance["depth"] == 0:
                instance["episodic_buffer_dict"] = {
                    "episodic_buffer_0": "This is a chaining task.",
                    "episodic_buffer_1": "Chunk 0 answers how many items each person had in the beginning.",
                    "episodic_buffer_2": "No one exchanged items with others.",
                    "episodic_buffer_3": f"How many {item_tuple[1]} did {main_chara} have in the end?",
                }
            else:
                instance["episodic_buffer_dict"] = {
                    "episodic_buffer_0": "This is a chaining task.",
                    "episodic_buffer_1": "Chunk 0 answers how many items each person had in the beginning.",
                    "episodic_buffer_2": (f"Chunk 1 to chunk {num_chains} can be used to infer "
                                          "how many items each person had after exchanging."),
                    "episodic_buffer_3": f"How many {item_tuple[1]} did {main_chara} have in the end?",
                }

        return instances


if __name__ == "__main__":
    GenerateEVRChainingData.debug_evr_instances()
