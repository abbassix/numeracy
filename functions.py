import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import tqdm


class CustomDataCollator(DataCollatorForLanguageModeling):

    mlm: bool = True
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary "
                "for masked language modeling. You should pass `mlm=False` to "
                "train on causal language modeling instead."
            )

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        NOTE: keep `special_tokens_mask` as an argument for avoiding error
        """

        # labels is batch_size x length of the sequence tensor
        # with the original token id
        # the length of the sequence includes the special tokens (2)
        labels = inputs.clone()

        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        # ==========================================================
        # the template is like this:
        # Q: What is 81 plus 14? A: 95
        # the result comes after `? a:` ([1029, 1037, 1024]) sequence
        # and before the [SEP] ([102]) token
        # here we find the indices of the result and mask them
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        masked_indices = torch.zeros_like(labels)
        for batch_number in range(batch_size):
            # loop from the end of the sequence to the beginning
            for i in range(seq_len - 1, 0, -1):
                # 102 stands for `[SEP]` token and comes after the result
                if labels[batch_number, i] == 102:
                    end_index = i
                # 1024 stands for `:` and the result comes after it
                if labels[batch_number, i] == 1024:
                    masked_indices[batch_number, i + 1:end_index] = 1
                    break
        masked_indices = masked_indices.bool()
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # We only compute loss on masked tokens
        labels[~masked_indices] = -100

        # Change the input's masked_indices to self.tokenizer.mask_token
        inputs[masked_indices] = self.tokenizer.mask_token_id

        return inputs, labels


def load_model(model_checkpoint: str, verbose=True):
    """
    Load the model and tokenizer from the local machine
    :param model_name: the name of the model
    :return: a tuple of the model and tokenizer
    """

    model_name = model_checkpoint.split("/")[-1]
    model_path = f"../models/{model_name}"
    try:
        # check if the model is in the local machine
        open(f"{model_path}/config.json")
        # load the model from the local machine
        print(f"\nLoading the model from the local machine: {model_path}")
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Model {model_name} is loaded from the local machine.")
    except FileNotFoundError:
        print(f"\nModel {model_name} is not found in the local machine.")
        print("Loading the model from the Hugging Face model hub!")
        # load the model from the Hugging Face model hub
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # save the model to the local machine
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    if verbose:
        num_parameters = model.num_parameters() / 1_000_000
        print(f">>> {model_name} is loaded.")
        print(f"The number of parameters: {round(num_parameters)}M")

    return model, tokenizer


def add_tokens(model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, new_tokens: list[str]):
    # check if the tokens are already in the vocabulary
    new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())

    # add the tokens to the tokenizer vocabulary
    tokenizer.add_tokens(list(new_tokens))

    # add new, random embeddings for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_dataset(dataset_name: str, verbose=True) -> Dataset:
    """
    Load the dataset from the local machine
    :param dataset_name: the name of the dataset
    :return: the dataset
    """

    # load the dataset from the local machine
    dataset = Dataset.load_from_disk(dataset_name)

    if verbose:
        # print the dataset is loaded
        print(f"\n>>> {dataset_name} is loaded.")

        # print number of rows in train, validation, and test
        # train_size = dataset['train'].num_rows
        # validation_size = dataset['validation'].num_rows
        # test_size = dataset['test'].num_rows
        # msg = "Number of rows in train, validation, and test:"
        # print(f"'{msg} {train_size}, {validation_size}, {test_size}'")

        # print the first row of the train dataset
        print(f"First row of the train dataset: {dataset[0]}")
        # print the size of the dataset
        print(f"Dataset has {len(dataset)} samples.")

    return dataset


def fine_tune(model: AutoModelForMaskedLM,
              tokenizer: AutoTokenizer,
              tokenized_dataset: Dataset,
              batch_size=32,
              num_epochs=1,
              lr=5e-5,
              collator="custom",
              weight_decay=0.0):
    """
    Fine-tune the model with the tokenized dataset
    :param model: the model
    :param tokenizer: the tokenizer
    :param dataset: the dataset
    :param masking_prob: the masking probability
    :param batch_size: the batch size
    :param num_epochs: the number of epochs
    :param lr: the learning rate
    :param weight_decay: the weight decay
    :return: the fine-tuned model
    """

    if collator == "custom":
        data_collator = CustomDataCollator(tokenizer=tokenizer)
    elif collator == "default":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)
    else:
        raise ValueError("Invalid collator!")

    # choose 10% of tokenized_dataset as the validation set
    tokenized_validation = tokenized_dataset.train_test_split(test_size=0.1)['test']

    training_args = TrainingArguments("test-clm",
                                      evaluation_strategy="epoch",
                                      learning_rate=lr,
                                      per_device_train_batch_size=batch_size,
                                      num_train_epochs=num_epochs,
                                      weight_decay=weight_decay)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=tokenized_dataset,
                      eval_dataset=tokenized_validation)

    trainer.train()

    return model


# to be edited
# mask the result of the arithmetic operation and check the accuracy
def accuracy(model: AutoModelForMaskedLM,
             tokenizer: AutoTokenizer,
             dataset: Dataset,
             verbose='') -> float:
    """
    Calculate the accuracy of the model on the test dataset.
    :param model: the model
    :param tokenizer: the tokenizer
    :param dataset: the dataset
    :param test_set: the test set
    :return: the accuracy
    """

    n_correct = 0

    if verbose == 'positive':
        pos_ = []
    elif verbose == 'negative':
        neg_ = []
    # loop through the test dataset
    # use tqdm to show the progress bar
    for input in tqdm.tqdm(dataset, desc="Calculating accuracy..."):
        input = input['text']
        input_ = input
        label = input
        # input: 'Q: What is 98 plus 45? A: 143'
        # or: 'Q: What is 9 8 plus 4 5? A: 1 4 3'
        # result comes after `? A:`. find the result
        result = input.split("? A: ")[1]
        input = input.split("? A: ")[0] + "? A: "
        # find out the number of tokens in the result
        n_tokens = len(tokenizer.tokenize(result))
        input = input + "[MASK] " * n_tokens

        input = tokenizer(input, return_tensors="pt")
        label = tokenizer(label, return_tensors="pt")
        # span of the masked token(s)
        masked_span = torch.where(input['input_ids'] == tokenizer.mask_token_id)
        outputs = model(**input)
        prediction = torch.argmax(outputs.logits, dim=-1)

        # check if the prediction is correct
        if torch.equal(prediction[0, masked_span[1]], label['input_ids'][0, masked_span[1]]):
            # print the decoded prediction
            pos_.append(input_) if verbose == 'positive' else None
            n_correct += 1
        else:
            # print the decoded prediction
            neg_.append(input_) if verbose == 'negative' else None
    if verbose == 'positive':
        print(f"Positive samples: {pos_}")
    elif verbose == 'negative':
        print(f"Negative samples: {neg_}")
    accuracy = 100 * n_correct / len(dataset)

    return accuracy
