import json
import numpy as np
import torch

from preliminary_experiments.utils.experiment_metric_utils import ExpMetricUtils
from preliminary_experiments.experiments_evr.evr_class.evr_backbone_lm import T5ModelEVR
from preliminary_experiments.utils.data_read_write_utils import DataReadWriteUtils


class EVRTrainer:

    def __init__(self,
                 training_config,
                 device,
                 exp_save_folder_path=None,
                 save_file_root_name=None,
                 print_every=200
                 ):

        self.training_config = training_config
        self.device = device

        # Define the most basic parameters
        self.machine_switch = training_config["machine_switch"]

        # Define model-related parameters
        self.model_name = training_config["model_name"]
        self.model_load_path = training_config["model_load_path"]
        self.transfer_model_data_n_amt = training_config["transfer_model_data_n_amt"]

        # The default learning rate (recommended) seems to be 1e-4
        # https://github.com/google-research/text-to-text-transfer-transformer/issues/230
        self.lr = training_config["lr"]
        self.t5model = T5ModelEVR(
            model_name=self.model_name,
            device=self.device,
            model_load_path=self.model_load_path,
        )
        self.optimizer = torch.optim.AdamW(params=self.t5model.model.parameters(), lr=self.lr)

        # Define experiment related parameters:
        self.n_train = training_config["n_train"]

        self.batch_size = training_config["batch_size"]
        self.grad_accu_num = training_config["grad_accu"]

        self.model_seed = training_config["model_seed"]

        self.n_epoch = training_config["n_epoch"]
        self.eval_every_k_batch = training_config["eval_every_k_batch"]

        self.patient_num = training_config["patient_num"]

        self.num_workers = training_config["num_workers"]

        self.print_every = print_every

        # Define save location
        self.exp_save_folder_path = exp_save_folder_path
        self.save_file_root_name = save_file_root_name

    def train_epoch(self, dataloader_train):
        self.t5model.model.train()

        total_loss = []
        for idx, batch in enumerate(dataloader_train):

            loss, prediction_scores = self.t5model.forward_batch_train(batch)

            # It is suggested by the online ref, it seems that the loss will be
            # automatically accumulated by the optimizer.
            # Finetuning t5 with batch size 32 or 64 gives the best result, so I use the gradient accumulation.
            loss = loss / self.grad_accu_num
            loss.backward()

            # This is for the gradient accumulation.
            # Source: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
            if (idx + 1) % self.grad_accu_num == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss.append(loss.detach().cpu().tolist())

            if (idx + 1) % self.print_every == 0:
                print("training loss:", np.mean(total_loss))
                total_loss = []

    def eval_epoch(self, dataloader_test, instances_test, debug_flag=False):
        eval_acc, input_text_all, pred_text_all, target_text_all, hit_by_task_n_depth, ids_all = \
            self.eval_epoch_public(self.t5model, dataloader_test, instances_test, debug_flag=debug_flag)

        return eval_acc, input_text_all, pred_text_all, target_text_all, hit_by_task_n_depth, ids_all

    @staticmethod
    def eval_epoch_public(t5model, dataloader_test, instances_test, print_every=20, debug_flag=False):

        t5model.model.eval()

        input_text_all = []
        pred_text_all = []
        target_text_all = []
        ids_all = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader_test):

                pred_texts, pred_tensors = t5model.forward_batch_eval(batch)

                input_text_all.extend(batch["input_text"])
                pred_text_all.extend(pred_texts)
                target_text_all.extend(batch["target_text"])
                ids_all.extend(batch["id"])

                if debug_flag:
                    print("-" * 40)
                    print(batch["task_pattern"][-1])
                    print("input:")
                    print(input_text_all[-1])
                    print("pred:")
                    print(pred_text_all[-1])
                    print("target")
                    print(target_text_all[-1])
                    print(pred_tensors.size())
                    print("hit?", ExpMetricUtils.get_evr_em(pred_text_all[-1], target_text_all[-1]))
                    input("-" * 40)

                if (idx + 1) % print_every == 0:
                    print("\tevaluating batch ", idx)

        hit_list = [
            ExpMetricUtils.get_evr_em(pred_text_all[i], target_text_all[i])
            for i in range(len(target_text_all))
        ]
        eval_acc = sum(hit_list) / len(hit_list)

        print("eval acc:", eval_acc)

        hit_by_task_n_depth = {}
        for i in range(len(instances_test)):

            task_pattern = instances_test[i]["task"] + "_" + str(instances_test[i]["pattern"])
            depth = instances_test[i]["depth"]

            if task_pattern not in hit_by_task_n_depth:
                hit_by_task_n_depth[task_pattern] = {}

            if depth not in hit_by_task_n_depth[task_pattern]:
                hit_by_task_n_depth[task_pattern][depth] = []

            hit_by_task_n_depth[task_pattern][depth].append(hit_list[i])

        print("acc by depth:")
        for t_p in sorted(hit_by_task_n_depth.keys()):
            for d in sorted(hit_by_task_n_depth[t_p].keys()):
                print("\t" + t_p + " depth:", d, " acc:", np.mean(hit_by_task_n_depth[t_p][d]))

        return eval_acc, input_text_all, pred_text_all, target_text_all, hit_by_task_n_depth, ids_all

    def train_and_eval(self,
                       dataloader_train,
                       dataloader_dev,
                       dataloader_test,
                       dataloader_test_ood,
                       instances_train,
                       instances_dev,
                       instances_test,
                       instances_test_ood,
                       debug_flag=False):

        best_dev_acc = 0
        test_ood_acc = 0
        metrics_to_save = {
            "training_config": self.training_config,
            "dev_acc": [],
            "test_acc": [],
            "test_ood_acc": [],
        }
        epoch_num = 0
        patient_counter = 0

        while epoch_num < self.n_epoch:

            print("=" * 40)
            print("train epoch ", epoch_num)

            self.t5model.model.train()
            total_loss = []
            for batch_iter, train_batch in enumerate(dataloader_train):

                if debug_flag:
                    print("-" * 40)
                    print(train_batch["input_text"])
                    print(train_batch["input"]["input_ids"])
                    print(train_batch["target_text"])
                    print(train_batch["target"]["input_ids"])
                    print([[self.t5model.tokenizer._convert_id_to_token(int(y)) for y in x if y != -100]
                           for x in train_batch["target"]["input_ids"]])
                    print([self.t5model.tokenizer.decode([y for y in x if y != -100]) for x in
                           train_batch["target"]["input_ids"]])
                    input("-" * 40)

                loss, prediction_scores = self.t5model.forward_batch_train(train_batch)

                # It is suggested by the online ref, it seems that the loss will
                # be automatically accumulated by the optimizer.
                # Finetuning t5 with batch size 32 or 64 gives the best result, so I use the gradient accumulation.
                loss = loss / self.grad_accu_num
                loss.backward()

                # This is for the gradient accumulation.
                # Source: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
                if (batch_iter + 1) % self.grad_accu_num == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss.append(loss.detach().cpu().tolist())

                if (batch_iter + 1) % self.print_every == 0:
                    print("training iter ", batch_iter + 1, " loss:", np.mean(total_loss))

                # Evaluate the model for certain steps of optimization or when this epoch is done.
                if (batch_iter + 1) % self.eval_every_k_batch == 0 or batch_iter == len(dataloader_train) - 1:
                    dev_acc, dev_input_text_all, dev_pred_text_all, dev_target_text_all, \
                        dev_hit_by_dpeth, dev_ids_all = \
                        self.eval_epoch(dataloader_dev, instances_dev, debug_flag=False)

                    if dev_acc > best_dev_acc:
                        best_dev_acc = dev_acc
                        torch.save(self.t5model.model, self.exp_save_folder_path + self.save_file_root_name)

                        test_ood_acc, test_ood_input_text_all, test_ood_pred_text_all, test_ood_target_text_all, \
                            test_ood_hit_by_depth, test_ood_ids_all = \
                            self.eval_epoch(dataloader_test_ood, instances_test_ood)

                        metrics_to_save["test_ood_pred_text_all"] = test_ood_pred_text_all
                        metrics_to_save["test_ood_target_text_all"] = test_ood_target_text_all
                        metrics_to_save["test_ood_hit_by_depth"] = test_ood_hit_by_depth
                        metrics_to_save["test_ood_ids_all"] = test_ood_ids_all

                        patient_counter = 0

                    else:
                        patient_counter += 1

                    metrics_to_save["dev_acc"].append(dev_acc)
                    metrics_to_save["test_ood_acc"].append(test_ood_acc)

                    self.t5model.model.train()

                    DataReadWriteUtils.write_json(
                        metrics_to_save,
                        self.exp_save_folder_path + "metrics_" + self.save_file_root_name + ".json"
                    )

                    if patient_counter >= self.patient_num or best_dev_acc > 1 - 1e-5:
                        break

            epoch_num += 1

            # If the dev acc has not been improved for 5 evaluations, break.
            if patient_counter >= self.patient_num or best_dev_acc > 1 - 1e-5:
                break
