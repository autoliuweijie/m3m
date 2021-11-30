# coding: utf-8
"""
Example of Bart model in transformers.

@ref: https://huggingface.co/transformers/v4.9.0/model_doc/bart.html
"""
from transformers import BartTokenizer, BartForConditionalGeneration


def summarization_example():
    model_path = '/root/data/private/models/huggingface_transformers/facebook/bart-base/'
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    summry = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    print(summry)


def mask_filling_example():
    model_path = '/root/data/private/models/huggingface_transformers/facebook/bart-base/'
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    TXT = "My friends are <mask> but they eat too many carbs."
    input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
    logits = model(input_ids).logits

    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predict_ids = probs.topk(5)

    predictions = tokenizer.decode(predict_ids).split()
    print(predictions)


if __name__ == "__main__":
    summarization_example()
    mask_filling_example()

