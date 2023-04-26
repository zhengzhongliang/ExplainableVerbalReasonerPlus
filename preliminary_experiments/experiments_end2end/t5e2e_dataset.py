from torch.utils.data import Dataset


class PadCollate:

    def __init__(self, tokenizer, max_len=1024, model_name="allenai/unifiedqa"):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name

    def pad_collate(self, batch):

        batch_to_return = {}
        if self.model_name.startswith("allenai/unifiedqa"):
            # Encoding of unifiedqa: https://github.com/allenai/unifiedqa
            batch_to_return["input_text"] = [sample["question_string"] + " \n " + sample["context_string_e2e"]
                                             for sample in batch]
            batch_to_return["input"] = self.tokenizer(batch_to_return["input_text"],
                                                      return_tensors="pt",
                                                      padding=True, truncation=True, max_length=self.max_len)

        else:  # should be t5 by default
            batch_to_return["input_text"] = [sample["context_string_e2e"] + " " + sample["question_string"]
                                             for sample in batch]
            batch_to_return["input"] = self.tokenizer(batch_to_return["input_text"],
                                                      return_tensors="pt",
                                                      padding=True, truncation=True, max_length=self.max_len)

        batch_to_return["target_text"] = [str(sample["answer"]) for sample in batch]
        batch_to_return["target"] = self.tokenizer(batch_to_return["target_text"],
                                                   return_tensors="pt",
                                                   padding=True, truncation=True, max_length=self.max_len)

        # Set the padding tokens to -100
        batch_to_return["target"]['input_ids'][
            batch_to_return["target"]['input_ids'] == self.tokenizer.pad_token_id] = -100

        batch_to_return["id"] = [sample["id"] for sample in batch]

        return batch_to_return

    def __call__(self, batch):
        return self.pad_collate(batch)


class DatasetT5E2E(Dataset):
    def __init__(self, instances):
        self.all_instances = instances

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, idx):
        return self.all_instances[idx]
