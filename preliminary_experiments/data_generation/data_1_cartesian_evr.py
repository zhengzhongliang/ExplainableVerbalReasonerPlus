import json

from preliminary_experiments.data_generation.data_utils import DataUtils


class GenerateEVRCartesianData:

    @classmethod
    def remove_string_leading_spaces(cls, input_string):

        start_idx = 0
        end_idx = len(input_string)

        while start_idx < len(input_string) and input_string[start_idx] == " ":
            start_idx += 1

        while end_idx > 0 and input_string[end_idx - 1] == " ":
            end_idx -= 1

        if start_idx > end_idx or input_string[start_idx: end_idx] == "":
            return None
        else:
            return input_string[start_idx: end_idx]

    @classmethod
    def get_item_by_quantity(cls, quantity, item_tuple):

        if int(quantity) == 1:
            return item_tuple[0]
        else:
            return item_tuple[1]

    @classmethod
    def get_persons(cls, instance):

        persons = []
        for t in instance["target_list"]:
            if t[0] not in persons:
                persons.append(t[0])
        return persons

    @classmethod
    def get_quant_items(cls, instance):

        quant_items = []
        for t in instance["target_list"]:
            if (t[1], t[2]) not in quant_items:
                quant_items.append((t[1], t[2]))
        return quant_items

    @classmethod
    def get_quant_items_nl(cls, instance):
        quant_items = cls.get_quant_items(instance)
        quant_items_nl = [f"{i_nl[0]} {cls.get_item_by_quantity(i_nl[0], i_nl[1])}" for i_nl in quant_items]
        return quant_items_nl

    @classmethod
    def generate_pattern_qa_1_data(cls, instance):

        evr_instances = []

        main_charas = cls.get_persons(instance)

        # input_text = (
        #     f"qa: {instance['context_string']} "
        #     "List the persons by copying the persons from the context."
        # )

        input_text = (
            f"qa: statement_0: {instance['context_string']} "
            "Who are the persons?"
        )

        # Build the list, and each name has surrounding single quotes
        main_charas_q = ["'" + m_c + "'" for m_c in main_charas]

        target_text = "The persons are [" + ", ".join(main_charas_q) + "]." \
            if len(main_charas_q) > 1 \
            else "The persons are [" + main_charas_q[0] + "]."

        evr_instance = {
            "task": "qa",
            "pattern": 1,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "main_charas": main_charas,
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_qa_2_data(cls, instance):

        evr_instances = []

        quant_items = cls.get_quant_items(instance)
        quant_items_nl = cls.get_quant_items_nl(instance)
        quant_items_nl_q = ["'" + q_i + "'" for q_i in quant_items_nl]

        # input_text = (
        #     f"qa: {instance['context_string']} "
        #     "List the items by copying the items from the context."
        # )

        input_text = (
            f"qa: statement_0: {instance['context_string']} "
            "What are the items?"
        )

        target_text = "The items are [" + ", ".join(quant_items_nl_q) + "]." \
            if len(quant_items_nl_q) > 1 \
            else "The items are [" + quant_items_nl_q[0] + "]."

        evr_instance = {
            "task": "qa",
            "pattern": 2,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
            "quant_items": quant_items,
            "quant_items_nl": quant_items_nl
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_1_data(cls, instance):

        evr_instances = []

        main_charas = []
        quant_items = []
        target_list = instance["target_list"]

        for target_statement in target_list:
            main_chara = target_statement[0]
            quant_item = (target_statement[1], target_statement[2])

            if main_chara not in main_charas:
                main_charas.append(main_chara)

            if quant_item not in quant_items:
                quant_items.append(quant_item)

        assert len(main_charas) == instance["depth"]
        assert len(quant_items) == instance["depth"]

        quant_items_nl = [
            str(quant_item[0]) + " " + cls.get_item_by_quantity(quant_item[0], quant_item[1])
            for quant_item in quant_items
        ]

        for main_chara in main_charas:
            for quant_item_nl in quant_items_nl:

                input_text = \
                    "rewrite: How many items did this person have? " + \
                    main_chara + ". " + quant_item_nl + "."

                target_text = main_chara + " had " + quant_item_nl + "."

                evr_instance = {
                    "task": "rewrite",
                    "pattern": 1,
                    "input": input_text,
                    "target": target_text,
                    "org_id": instance["id"],
                    "depth": instance["depth"],
                }

                evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_rewrite_2_data(cls, instance):

        evr_instances = []

        input_list = instance["answer"].replace(".", "").split(",")
        input_list = [cls.remove_string_leading_spaces(x) for x in input_list]
        input_list = [x + "." for x in input_list if not x.isspace() and x != ""]

        input_text = " ".join(input_list)
        input_text = ("rewrite: "
                      "Change the list to a natural language sentence. "
                      f"{input_text}")

        target_text = instance["answer"]

        evr_instance = {
            "task": "rewrite",
            "pattern": 2,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_clear_mem_1_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        instance_items_nl = cls.get_quant_items_nl(instance)
        quant_items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        quant_items_list_str = f"[{quant_items_list_str}]"

        evr_instances = []

        # Prepare the input text
        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The persons are {persons_list_str}.",
            f"The items are {quant_items_list_str}.",
            "#0 stores the list of persons."
        ]
        input_text = " ".join([f"episodic_buffer_{idx}: {i_t}" for idx, i_t in enumerate(input_text_list)])
        input_text = f"clear_mem: {input_text}"

        target_text_list = [
            "'This is a cartesian task.'",
            "'Chunk 0 can be used to infer the number of items each person had.'",
            "'List the items that each person had.'",
            f"'The items are {quant_items_list_str}.'",
            "'#0 stores the list of persons.'"
        ]
        target_text = " ".join([f"episodic_buffer_{idx}: {i_t}" for idx, i_t in enumerate(target_text_list)])

        evr_instance = {
            "task": "clear_mem",
            "pattern": 1,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_clear_mem_2_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        instance_items_nl = cls.get_quant_items_nl(instance)
        quant_items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        quant_items_list_str = f"[{quant_items_list_str}]"

        evr_instances = []

        # Prepare the input text
        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The items are {quant_items_list_str}.",
            "#0 stores the list of persons.",
            "#1 stores the list of items."
        ]
        input_text = " ".join([f"episodic_buffer_{idx}: {i_t}" for idx, i_t in enumerate(input_text_list)])
        input_text = f"clear_mem: {input_text}"

        target_text_list = [
            "'This is a cartesian task.'",
            "'Chunk 0 can be used to infer the number of items each person had.'",
            "'List the items that each person had.'",
            "'#0 stores the list of persons.'",
            "'#1 stores the list of items.'"
        ]
        target_text = " ".join([f"episodic_buffer_{idx}: {i_t}" for idx, i_t in enumerate(target_text_list)])

        evr_instance = {
            "task": "clear_mem",
            "pattern": 2,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_1_data(cls, instance):

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "#0 = 'Who are the persons?';",
            "new_mem(episodic_buffer_1, #0);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 1,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_2_data(cls, instance):

        evr_instances = []

        input_text_list = [
            "Chunk 0 can be used to infer the number of items each person had.",
            "Who are the persons?",
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "#0 = get_chunk('chunk_0');",
            "#1 = qa(#0, episodic_buffer_1);",
            "add_to_episodic(#1);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 2,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_3_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        evr_instances = []

        input_text_list = [
            "Chunk 0 can be used to infer the number of items each person had.",
            "Who are the persons?",
            f"The persons are {persons_list_str}."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "return(episodic_buffer_2);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 3,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_4_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The persons are {persons_list_str}."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "#0 = 'What are the items?';",
            "new_mem(episodic_buffer_1, #0);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 4,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_5_data(cls, instance):

        evr_instances = []

        input_text_list = [
            "Chunk 0 can be used to infer the number of items each person had.",
            "What are the items?",
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "#0 = get_chunk('chunk_0');",
            "#1 = qa(#0, episodic_buffer_1);",
            "add_to_episodic(#1);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 5,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_6_data(cls, instance):

        instance_items_nl = cls.get_quant_items_nl(instance)
        items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        items_list_str = f"[{items_list_str}]"

        evr_instances = []

        input_text_list = [
            "Chunk 0 can be used to infer the number of items each person had.",
            "What are the items?",
            f"The items are {items_list_str}."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "return(episodic_buffer_2);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 6,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_7_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        instance_items_nl = cls.get_quant_items_nl(instance)
        quant_items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        quant_items_list_str = f"[{quant_items_list_str}]"

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The persons are {persons_list_str}.",
            f"The items are {quant_items_list_str}."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            f"#0 = {persons_list_str};",
            "add_to_episodic('#0 stores the list of persons.');"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 7,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_8_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        instance_items_nl = cls.get_quant_items_nl(instance)
        quant_items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        quant_items_list_str = f"[{quant_items_list_str}]"

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The persons are {persons_list_str}.",
            f"The items are {quant_items_list_str}.",
            "#0 stores the list of persons."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "clear_mem();"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 8,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_9_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        instance_items_nl = cls.get_quant_items_nl(instance)
        quant_items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        quant_items_list_str = f"[{quant_items_list_str}]"

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The items are {quant_items_list_str}.",
            "#0 stores the list of persons."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            f"#1 = {quant_items_list_str};",
            "add_to_episodic('#1 stores the list of items.');"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 9,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_10_data(cls, instance):

        instance_persons = cls.get_persons(instance)
        persons_list_str = ", ".join([f"'{p}'" for p in instance_persons])
        persons_list_str = f"[{persons_list_str}]"

        instance_items_nl = cls.get_quant_items_nl(instance)
        quant_items_list_str = ", ".join([f"'{i_nl}'" for i_nl in instance_items_nl])
        quant_items_list_str = f"[{quant_items_list_str}]"

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            f"The items are {quant_items_list_str}.",
            "#0 stores the list of persons.",
            "#1 stores the list of items."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "clear_mem();"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 10,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_11_data(cls, instance):

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            "#0 stores the list of persons.",
            "#1 stores the list of items."
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "#2 = [];",
            "for #3 in #0;",
            "for #4 in #1;",
            "#5 = 'How many items did this person have?';",
            "#6 = rewrite(#5, #3, #4);",
            "#2 = append_to_list(#2, #6);",
            "end_for;",
            "end_for;",
            #"#7 = rewrite('Change the list to a natural language sentence', #2);",
            "return(#2);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 11,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_pattern_gen_prog_12_data(cls, instance):

        evr_instances = []

        input_text_list = [
            "This is a cartesian task.",
            "Chunk 0 can be used to infer the number of items each person had.",
            "List the items that each person had.",
            "#0 stores the list of persons.",
            "#1 stores the list of items.",
            instance["answer"]
        ]
        input_text = " ".join([f"episodic_buffer_{e_idx}: {i_t}" for e_idx, i_t in enumerate(input_text_list)])
        input_text = f"generate_program: {input_text}"

        # Prepare the target text
        target_text_list = [
            "return(episodic_buffer_5);"
        ]
        target_text = " ".join(target_text_list)

        evr_instance = {
            "task": "generate_program",
            "pattern": 12,
            "input": input_text,
            "target": target_text,
            "org_id": instance["id"],
            "depth": instance["depth"],
        }

        evr_instances.append(evr_instance)

        return evr_instances

    @classmethod
    def generate_evr_data_one_instance(cls, instance):

        pattern_generation_func = {
            "qa": {
                1: cls.generate_pattern_qa_1_data,
                2: cls.generate_pattern_qa_2_data
            },
            "rewrite": {
                1: cls.generate_pattern_rewrite_1_data,
                #2: cls.generate_pattern_rewrite_2_data
            },
            "clear_mem": {
                1: cls.generate_pattern_clear_mem_1_data,
                2: cls.generate_pattern_clear_mem_2_data
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
                #12: cls.generate_pattern_gen_prog_12_data,
            },
        }

        evr_instances = []

        for pattern in pattern_generation_func.keys():
            for pattern_num in pattern_generation_func[pattern].keys():
                evr_instances_one_pattern = pattern_generation_func[pattern][pattern_num](instance)
                evr_instances.extend(evr_instances_one_pattern)

        return evr_instances

    @classmethod
    def generate_evr_instances(cls, instances):

        evr_instances_all_patterns = []

        for instance in instances:
            evr_instances = cls.generate_evr_data_one_instance(instance)
            evr_instances_all_patterns.extend(evr_instances)

        return evr_instances_all_patterns

    @classmethod
    def debug_evr_instances(cls):

        instances = DataUtils.load_json(
            "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/"
            "data_generated/cartesian_v0.1/cartesian_data_du2.json"
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
            instance["external_chunks"] = {
                "chunk_0": instance["context_string"]
            }

            instance["episodic_buffer_dict"] = {
                "episodic_buffer_0": "This is a cartesian task.",
                "episodic_buffer_1": "Chunk 0 can be used to infer the number of items each person had.",
                "episodic_buffer_2": instance["question_string"]
            }

        return instances


if __name__ == "__main__":
    GenerateEVRCartesianData.debug_evr_instances()
