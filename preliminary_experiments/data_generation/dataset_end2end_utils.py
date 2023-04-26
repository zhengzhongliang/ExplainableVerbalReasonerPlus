import math


class DatasetEnd2endUtils:

    @classmethod
    def get_chaining_example_chunks(cls, chaining_instance):

        chains = chaining_instance["chains"]
        n_chains = len(chains)

        beginning_statements = " ".join([c["context_list"][0] for c in chains])
        if chaining_instance["depth"] > 0:
            exchanging_statements = [" ".join(c["context_list"][1:]) for c in chains]
        else:
            exchanging_statements = []

        chunks_list = [beginning_statements] + exchanging_statements

        return chunks_list

    @classmethod
    def get_tree_search_example_chunks(cls, context_list, chunk_size):

        n_chunk = math.ceil(len(context_list) / 3)
        chunks_list = []
        for chk_idx in range(n_chunk):
            chunks_list.append(context_list[chk_idx * chunk_size: (chk_idx + 1) * chunk_size])
        chunks_list = [" ".join(c) for c in chunks_list]

        return chunks_list

    @classmethod
    def convert_chaining_examples(cls, instances):

        for split in ["train", "dev", "test"]:
            for instance in instances[split]:
                chunks_list = cls.get_chaining_example_chunks(instance)
                chunks_list = [f"chunk_{idx_}: {c_t}" for idx_, c_t in enumerate(chunks_list)]

                n_chains = len(instance["chains"])

                if instance["depth"] > 0:
                    prompt_list = [
                        "This is a chaining task.",
                        "Chunk 0 answers how many items each person had in the beginning.",
                        (f"Chunk 1 to chunk {n_chains} can be used to infer "
                         "how many items each person had after exchanging.")
                    ]
                else:
                    prompt_list = [
                        "This is a chaining task.",
                        "Chunk 0 answers how many items each person had in the beginning.",
                        "No one exchanged items with other."
                    ]

                context_list = prompt_list + chunks_list
                context_string = " ".join(context_list)

                instance["context_string_e2e"] = context_string

        return instances

    @classmethod
    def convert_cartesian_examples(cls, instances):

        for split in ["train", "dev", "test"]:
            for instance in instances[split]:
                prompt_list = [
                    "This is a cartesian task.",
                    "Chunk 0 can be used to infer the number of items each person had.",
                ]

                chunk_list = [
                    f"chunk_0: {instance['context_string']}"
                ]

                context_list = prompt_list + chunk_list
                context_string = " ".join(context_list)

                instance["context_string_e2e"] = context_string

        return instances

    @classmethod
    def convert_tree_search_examples(cls, instances):

        for split in ["train", "dev", "test"]:
            for instance in instances[split]:
                chunks_list = instance["context_list"]

                prompt_list = [
                    "This is a tree search task.",
                ]

                context_list = prompt_list + chunks_list
                context_string = " ".join(context_list)

                instance["context_string_e2e"] = context_string

        return instances

    @classmethod
    def convert_chaining_tree_search_examples(cls, instances):
        for split in ["train", "dev", "test"]:
            for instance in instances[split]:

                tr_rules_list = [r for r in instance["tree_search_instance"]["context_list"] if r.startswith("If")]

                chaining_chunks_list = cls.get_chaining_example_chunks(instance["chaining_instance"])
                chaining_chunks_list = [f"chunk_{idx_}: {c_t}" for idx_, c_t in enumerate(chaining_chunks_list)]
                chunks_list = chaining_chunks_list + tr_rules_list

                n_chains = len(instance["chaining_instance"]["chains"])

                if instance["depth"] > 0:
                    prompt_list = [
                        "This is a chaining tree search task.",
                        "Chunk 0 answers how many items each person had in the beginning.",
                        (f"Chunk 1 to chunk {n_chains} can be used to infer "
                         "how many items each person had after exchanging.")
                    ]
                else:
                    prompt_list = [
                        "This is a chaining tree search task.",
                        "Chunk 0 answers how many items each person had in the beginning.",
                        "No one exchanged items with others."
                    ]

                context_list = prompt_list + chunks_list
                context_string = " ".join(context_list)

                instance["context_string_e2e"] = context_string

        return instances

    @classmethod
    def convert_cartesian_tree_search_examples(cls, instances):

        for split in ["train", "dev", "test"]:
            for instance in instances[split]:
                tr_rules_list = [r for r in instance["tree_search_instance"]["context_list"] if r.startswith("If")]

                cartesian_chunks_list = [f"chunk_0: {instance['cartesian_instance']['context_string']}"]
                chunks_list = cartesian_chunks_list + tr_rules_list

                prompt_list = [
                    "This is a cartesian tree search task.",
                    "Chunk 0 can be used to infer the number of items each person had."
                ]

                context_list = prompt_list + chunks_list
                context_string = " ".join(context_list)

                instance["context_string_e2e"] = context_string

        return instances

    @classmethod
    def convert_instances(cls, instances_all_depth, task_name, chunk_size=3):
        for du in instances_all_depth:
            if task_name.startswith("chaining_tree_search"):
                instances_all_depth[du] = cls.convert_chaining_tree_search_examples(
                    instances_all_depth[du]
                )
            elif task_name.startswith("cartesian_tree_search"):
                instances_all_depth[du] = cls.convert_cartesian_tree_search_examples(
                    instances_all_depth[du]
                )
            elif task_name.startswith("chaining"):
                instances_all_depth[du] = cls.convert_chaining_examples(
                    instances_all_depth[du]
                )
            elif task_name.startswith("cartesian"):
                instances_all_depth[du] = cls.convert_cartesian_examples(
                    instances_all_depth[du]
                )
            elif task_name.startswith("tree_search"):
                instances_all_depth[du] = cls.convert_tree_search_examples(
                    instances_all_depth[du]
                )
            else:
                raise Exception

        return instances_all_depth

