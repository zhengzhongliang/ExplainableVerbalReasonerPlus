import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5Model:

    '''
    Typical learning rate and batch size: 1e-4, 3e-4 and 16, 32.
    Ref 1 (3e-4, 16): https://huggingface.co/deep-learning-analytics/wikihow-t5-small
    Ref 2 (1e-4 and 3e-4): https://huggingface.co/docs/transformers/model_doc/t5
    '''

    def __init__(self,
                 device,
                 model_name,
                 model_load_path,
                 transfer_model_data_n_amt,
                 model_gen_len=400):

        self.model_load_path = model_load_path
        self.model_name = model_name
        self.transfer_model_data_n_amt = transfer_model_data_n_amt

        if self.model_load_path == "None":
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        else:
            self.model = torch.load(self.model_load_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.device = device
        self.max_gen_len = model_gen_len

        self.model.to(device)

        if self.model_load_path == "None":
            print("Loaded " + self.model_name + " from scratch")
        else:
            print("Loaded model from:", self.model_load_path)
            print("Transfer model name:", self.transfer_model_data_n_amt)

    def forward_batch_train(self, batch):
        '''
        takes a batch, returns loss
        :return:
        '''

        outputs = self.model(input_ids=batch["input"]["input_ids"].to(self.device),
                             attention_mask=batch["input"]["attention_mask"].to(self.device),
                             labels=batch["target"]["input_ids"].to(self.device))
        loss, prediction_scores = outputs.loss, outputs.logits

        return loss, prediction_scores

    def forward_batch_eval(self, batch):
        '''
        takes a batch, returns the decoded text
        :return:
        '''

        pred_tensors = self.model.generate(input_ids=batch["input"]["input_ids"].to(self.device),
                                           attention_mask=batch["input"]["attention_mask"].to(self.device),
                                           max_length=self.max_gen_len)

        # Skip the special tokens:
        # https://huggingface.co/docs/transformers/internal/tokenization_utils
        pred_texts = self.tokenizer.batch_decode(pred_tensors, skip_special_tokens=True)

        return pred_texts, pred_tensors
