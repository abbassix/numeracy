import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import inflect


# receive input and model path from args
parser = argparse.ArgumentParser(description="Load configuration variables from command-line arguments")
parser.add_argument('--model', type=str, required=True, help='Path to the model')
parser.add_argument('--dataset', type=str, required=True, help='Input dataset')
parser.add_argument('--word', action='store_true', help='Print verbose output')
parser.add_argument('--verbose', action='store_true', help='Print verbose output')

args = parser.parse_args()

model_path = args.model
dataset_path = args.dataset
word = args.word
verbose = args.verbose

# load bert tokenizer from local machine
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


with open(dataset_path, 'r') as f:
    problems = f.readlines()

n_correct = 0
diff = 0
diff_list = []
rel_diff = 0
rel_diff_list = []
n_skipped = 0
for problem in problems:
    q, a = problem.split('Answer:')[0].rstrip(), problem.split('Answer:')[1].lstrip().rstrip()
    input = f"{q} Answer: [MASK]"
    if word:
        p = inflect.engine()
        # a = p.number_to_words(a)
        # find all numbers in the question and convert them to words
        q = q.replace('.', ' .')
        for word in q.split():
            try:
                int(word)
            except:
                continue
            q = q.replace(word, p.number_to_words(word))
        q = q.replace(' .', '.')
    input = tokenizer(input, return_tensors="pt")
    masked_span = torch.where(input['input_ids'] == tokenizer.mask_token_id)
    outputs = model(**input)
    prediction = torch.argmax(outputs.logits, dim=-1)

    pred = tokenizer.decode(prediction[0, masked_span[1]])
    if pred == a:
        print(f"Question: {q}\nAnswer: {a}\nPrediction: {pred}\n\n") if verbose else None
        n_correct += 1
    elif not word:
        # if the prediction cannot be converted to an integer, skip
        try:
            int(pred)
        except:
            n_skipped += 1
            continue
        diff_list.append(abs(int(a) - int(pred)))
        rel_diff_list.append(abs(int(a) - int(pred))/int(a))
        diff += abs(int(a) - int(pred))
        rel_diff += abs(int(a) - int(pred))/int(a)
    # print(f"Question: {q}\nAnswer: {a}\nPrediction: {pred}\n\n") if verbose else None

# calculate the frequency of differences
diff_list.sort()
rel_diff_list.sort()
freq = {}
for i in diff_list:
    if i not in freq:
        freq[i] = 1
    else:
        freq[i] += 1
rel_freq = {}
for i in rel_diff_list:
    if i not in rel_freq:
        rel_freq[i] = 1
    else:
        rel_freq[i] += 1
print(f"Accuracy: {n_correct/len(problems)}")
print(f"Average difference: {diff/(len(problems)-n_skipped)}")
print(f"Average relative difference: {rel_diff/(len(problems)-n_skipped)}")
# show frequency of differences
print(f"Frequency of differences: {freq}")
print(f"Frequency of relative differences: {rel_freq}")
