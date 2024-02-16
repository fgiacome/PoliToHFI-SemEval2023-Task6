from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizerFast
import torch

# tokenizer will be initialized later, but it is declared as
# a global variable for easy access by the `tokenize` function
tokenizer: XLMRobertaTokenizerFast = None
def tokenize(sample):
    # tokenize the text (this will create a dictionary with 'input_ids'
    # (tokens) and 'attention_mask')
    tokenized_text = tokenizer(sample['tokens'],
                            truncation=True,
                            verbose=False,
                            padding='max_length',
                            is_split_into_words=True)
    # The word_labels is a list of (numerical) tags for the words.
    # Remember, the input is already tokenized into words, so the
    # origial text is a list (but we need to have it processed by
    # our own tokenizer before feeding it into the model).
    word_labels = sample['ner_tags']
    # the following list will be used to store the tags for the
    # tokens produced by our tokenizer
    token_labels = [38]*len(tokenized_text['input_ids'])
    # A trick: since subsequent tokens can refer to the same word,
    # we have to remember whether a new label was already started,
    # otherwise we might end up with several 'B' tokens in a row
    # that refer to the same label.
    last_b_word_i = None
    for t_i in range(len(token_labels)):
        w_i = tokenized_text.token_to_word(t_i)
        if w_i is not None and token_labels[t_i] == 38:
            offset = 0
            if last_b_word_i == w_i:
                offset = 19
            token_labels[t_i] = word_labels[w_i] + offset
            if word_labels[w_i] < 19:
                last_b_word_i = w_i
    # store the labels in the output
    tokenized_text['labels'] = token_labels
    return tokenized_text


def get_german_dataset(split="train"):
    global tokenizer
    # initialize the tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base',
                                                        add_prefix_space=True)
    dataset = load_dataset("elenanereiss/german-ler", split=split)
    dataset = dataset.map(tokenize, batched = False)
    dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "labels"])
    return dataset
