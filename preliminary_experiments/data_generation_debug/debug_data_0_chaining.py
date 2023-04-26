import os
import json

from preliminary_experiments.data_generation.data_0_chaining import GenerateChainingData
from preliminary_experiments.data_generation.data_utils import DataUtils
from preliminary_experiments.data_generation.dataset_utils import ExpDatasetUtils


class GenerateChainingDataDebug(GenerateChainingData):

    @classmethod
    def chaining_translate_formal_rep(cls, main_chara, item_quantity, item):

        if item_quantity == 0:
            statement = main_chara + " had no " + item[0] + "."
        elif item_quantity == 1:
            statement = main_chara + " had 1 " + item[0] + "."
        else:
            statement = main_chara + " had " + str(item_quantity) + " " + item[1] + "."

        return statement

    @classmethod
    def debug_generate_training_data_chaining(cls, machine_switch="mac", debug_flag=False, debug_train_du=2):
        """
        This function is to generate some dummy debugging data, so that we can make sure 500 training data is enough
        for EVR to learn robust and correct patterns.
        :return:
        """

        # The seed and n_train should be hard coded. This way we use all of the raw training data.
        instances_all_depth = ExpDatasetUtils.load_data(seed=0,
                                                        n_train=20000,
                                                        machine_switch=machine_switch,
                                                        data_pattern="chaining_v0.4",
                                                        dev_ratio=0.1)

        train_dus = [2, 5]
        splits = ["train", "dev", "test"]

        instances_evr = {du: {split: [] for split in splits} for du in train_dus}
        for train_du in train_dus:
            for split in splits:
                for inst_idx, instance in enumerate(instances_all_depth[train_du][split]):
                    chaining_evr_train = cls.debug_generate_training_data_chaining_one_instance(instance)
                    instances_evr[train_du][split].append(chaining_evr_train)

            data_folder_dir = ("/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/"
                               "data_generated/chaining_v0.4_evr_v0.1/")

            if not os.path.exists(data_folder_dir):
                os.mkdir(data_folder_dir)

            with open(data_folder_dir + "chaining_v0.4_evr_v0.1_du" + str(train_du) + ".json", "w") as handle:

                json.dump(instances_evr[train_du], handle)

        if debug_flag:

            for split in splits:

                for inst_idx, instance in enumerate(instances_all_depth[debug_train_du][split]):

                    selected_chain_idx = instance["selected_chain_idx"]
                    selected_chain = instance["chains"][selected_chain_idx]
                    chaining_evr_train = instances_evr[debug_train_du][split][inst_idx]

                    print("=" * 40)

                    print("-" * 40)
                    for ctx in selected_chain["context_list"]:
                        print("\t", ctx)

                    print("-" * 40)
                    for evr_step_pair in chaining_evr_train:
                        print(evr_step_pair["input"])
                        print(evr_step_pair["target"])
                        print("\n")

                    input("-" * 40)

    @classmethod
    def debug_generate_training_data_chaining_one_instance(cls, chaining_instance):

        # TODO: need to handle depth 0 data?

        selected_chain_idx = chaining_instance["selected_chain_idx"]
        selected_chain = chaining_instance["chains"][selected_chain_idx]

        train_chaining_instances = []
        buffer_val = selected_chain["quantity_ops"][0]
        main_chara = selected_chain["formal_reps"][0][0]
        item = selected_chain["formal_reps"][0][2]

        for step_idx in range(1, len(selected_chain["context_list"])):
            buffer_val = buffer_val + selected_chain["quantity_ops"][step_idx]
            if step_idx == 1:
                one_step_training_instance = {
                    "input": selected_chain["context_list"][0] + " " + selected_chain["context_list"][1],
                    "target": cls.chaining_translate_formal_rep(main_chara, buffer_val, item),
                    "step_buffer_val": buffer_val
                }
            else:
                one_step_training_instance = {
                    "input": train_chaining_instances[-1]["target"] + " " + selected_chain["context_list"][step_idx],
                    "target": cls.chaining_translate_formal_rep(main_chara, buffer_val, item),
                    "step_buffer_val": buffer_val
                }

            train_chaining_instances.append(one_step_training_instance)

        assert buffer_val == selected_chain["answer"]

        return train_chaining_instances

    @classmethod
    def debug_data_generation_of_certain_depth(cls, debug_flag=False):

        splits = ["train", "dev", "test"]

        dataset_statistics = {
            split: {
                "beginning_num_distribution": {},
                "op_num_distribution": {},
                "answer_num_distribution": {}
            } for split in splits
        }

        dataset_depth = 0
        n_train = 10000
        n_dev = 2000
        n_test = 2000
        instances_all_splits = GenerateChainingData.generate_data_with_certain_depth(
            dataset_depth, 10000, 2000, 2000, debug_flag)

        for split in splits:
            for inst_idx, instance in enumerate(instances_all_splits[split]):
                # Get the distribution of the answers
                if instance["answer"] not in dataset_statistics[split]["answer_num_distribution"]:
                    dataset_statistics[split]["answer_num_distribution"][instance["answer"]] = 1
                else:
                    dataset_statistics[split]["answer_num_distribution"][instance["answer"]] += 1

                for chain in instance["chains"]:
                    # Get the distribution of the beginning numbers
                    if chain["quantity_ops"][0] not in dataset_statistics[split]["beginning_num_distribution"]:
                        dataset_statistics[split]["beginning_num_distribution"][chain["quantity_ops"][0]] = 1
                    else:
                        dataset_statistics[split]["beginning_num_distribution"][chain["quantity_ops"][0]] += 1

                    # Get the distribution of the sampled operations
                    sample_ops = chain["quantity_ops"][1:]
                    for sample_op in sample_ops:
                        if sample_op not in dataset_statistics[split]["op_num_distribution"]:
                            dataset_statistics[split]["op_num_distribution"][sample_op] = 1
                        else:
                            dataset_statistics[split]["op_num_distribution"][sample_op] += 1

            # Sort the distributions by keys
            dataset_statistics[split]["beginning_num_distribution"] = \
                {x[0]: x[1] for x in
                 sorted(dataset_statistics[split]["beginning_num_distribution"].items(), key=lambda x: x[0])}
            dataset_statistics[split]["op_num_distribution"] = \
                {x[0]: x[1] for x in
                 sorted(dataset_statistics[split]["op_num_distribution"].items(), key=lambda x: x[0])}
            dataset_statistics[split]["answer_num_distribution"] = \
                {x[0]: x[1] for x in
                 sorted(dataset_statistics[split]["answer_num_distribution"].items(), key=lambda x: x[0])}

            print("=" * 40)
            print(split)
            print(dataset_statistics[split])

    @classmethod
    def debug_check_one_generated_example(cls, du=2, split="train", depth=2):

        data_path = ("/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/"
                     "chaining_v0.4/chaining_data_du" + str(du) + ".json")

        instances_all_splits = DataUtils.load_json(data_path)

        for instance in instances_all_splits[split]:
            if instance["depth"] == depth:
                print("=" * 40)
                print(instance["context_string"])
                print(instance["question_string"])
                print(instance["answer"])

                print("depth:", instance["depth"])

                input("-" * 40)

    @classmethod
    def debug_generate_one_instance(cls, depth=2):

        for i in range(100):
            instance = GenerateChainingData.generate_one_example(
                depth=depth, num_chains=None,
                initial_statements=set([]),
                existing_grounded_chara_item=set([]),
                existing_grounded_chara_quant_item=set([]),
                existing_ungrounded_chara_quant_item=set([]))

            print("=" * 40)
            print(json.dumps(instance, indent=2))
            input("-" * 20)


if __name__ == "__main__":
    GenerateChainingDataDebug.debug_generate_one_instance(depth=4)
