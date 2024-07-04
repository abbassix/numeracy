import jsonlines
import re
import requests
import random
from datasets import Dataset
import argparse
import inflect


def generate_pool(n_digits=[2, 3, 4, 5], n_samples=3000):
    # set the seed for reproducibility
    random.seed(42)

    pool_ = {'addition': [], 'subtraction': [], 'multiplication': []}
    for n_digit in n_digits:
        if n_digit == 2:
            operations = [('addition', '+'), ('subtraction', '-'), ('multiplication', '*')]
            range_min, range_max = 0, 100
        else:
            operations = [('addition', '+'), ('subtraction', '-')]
            range_min, range_max = 10 ** (n_digit - 1), 10 ** n_digit
        for operation in operations:
            for _ in range(n_samples):
                pool_[operation[0]].append((random.randint(range_min, range_max), random.randint(range_min, range_max), operation[1]))

    # shuffle the pool
    for key in pool_.keys():
        random.shuffle(pool_[key])

    return pool_


def read_dataset():
    """
    Read the dataset from the GitHub repository
    :return: the dataset
    """
    dataset = dict()
    files = ['two_digit_addition.jsonl', 'two_digit_subtraction.jsonl', 'two_digit_multiplication.jsonl',
             'three_digit_addition.jsonl', 'three_digit_subtraction.jsonl',
             'four_digit_addition.jsonl', 'four_digit_subtraction.jsonl',
             'five_digit_addition.jsonl', 'five_digit_subtraction.jsonl',
             'six_digit_addition.jsonl', 'six_digit_subtraction.jsonl']

    for file in files:
        print(f'Reading file: {file}')
        operation = ''
        if 'addition' in file:
            operation = '+'
        elif 'subtraction' in file:
            operation = '-'
        elif 'multiplication' in file:
            operation = '*'
        dataset[file.split('.')[0]] = []
        url = f'https://raw.githubusercontent.com/openai/gpt-3/master/data/{file}'
        response = requests.get(url)
        with jsonlines.Reader(response.text.splitlines()) as reader:
            for line in reader:
                a = re.findall(r'\d+', line['context'])[0]
                b = re.findall(r'\d+', line['context'])[1]
                dataset[file.split('.')[0]].append((a, b, operation))

    return dataset


def baseline_addition(a, b):
    sum = a + b
    return f'Q: What is {a} plus {b}? A: {sum}'


def spaced_addition(a, b):
    a_spcd = ' '.join(list(str(a)))
    b_spcd = ' '.join(list(str(b)))
    sum_spcd = ' '.join(list(str(a + b)))
    return f'Q: What is {a_spcd} plus {b_spcd}? A: {sum_spcd}'


def pn_addition(a, b):
    a_pn = '[D] ' + ' '.join(list(str(a))) + ' [D]'
    b_pn = '[D] ' + ' '.join(list(str(b))) + ' [D]'
    sum_pn = '[D] ' + ' '.join(list(str(a + b))) + ' [D]'
    return f'Q: What is {a_pn} plus {b_pn}? A: {sum_pn}'


def baseline_subtraction(a, b):
    dif = a - b
    return f'Q: What is {a} minus {b}? A: {dif}'


def spaced_subtraction(a, b):
    a_spcd = ' '.join(list(str(a)))
    b_spcd = ' '.join(list(str(b)))
    dif_spcd = ' '.join(list(str(a - b)))
    return f'Q: What is {a_spcd} minus {b_spcd}? A: {dif_spcd}'


def pn_subtraction(a, b):
    a_pn = '[D] ' + ' '.join(list(str(a))) + ' [D]'
    b_pn = '[D] ' + ' '.join(list(str(b))) + ' [D]'
    dif_pn = '[D] ' + ' '.join(list(str(a - b))) + ' [D]'
    return f'Q: What is {a_pn} minus {b_pn}? A: {dif_pn}'


def baseline_multiplication(a, b):
    mul = a * b
    return f'Q: What is {a} times {b}? A: {mul}'


def spaced_multiplication(a, b):
    a_spcd = ' '.join(list(str(a)))
    b_spcd = ' '.join(list(str(b)))
    mul_spcd = ' '.join(list(str(a * b)))
    return f'Q: What is {a_spcd} times {b_spcd}? A: {mul_spcd}'


def pn_multiplication(a, b):
    a_pn = '[D] ' + ' '.join(list(str(a))) + ' [D]'
    b_pn = '[D] ' + ' '.join(list(str(b))) + ' [D]'
    mul_pn = '[D] ' + ' '.join(list(str(a * b))) + ' [D]'
    return f'Q: What is {a_pn} times {b_pn}? A: {mul_pn}'


def baseline(a, b, operation):
    if operation == '+':
        return baseline_addition(a, b)
    elif operation == '-':
        return baseline_subtraction(a, b)
    elif operation == '*':
        return baseline_multiplication(a, b)
    else:
        raise ValueError("Invalid type!")


def spaced(a, b, operation):
    if operation == '+':
        return spaced_addition(a, b)
    elif operation == '-':
        return spaced_subtraction(a, b)
    elif operation == '*':
        return spaced_multiplication(a, b)
    else:
        raise ValueError("Invalid type!")


def pn(a, b, operation):
    if operation == '+':
        return pn_addition(a, b)
    elif operation == '-':
        return pn_subtraction(a, b)
    elif operation == '*':
        return pn_multiplication(a, b)
    else:
        raise ValueError("Invalid type!")


def save_dataset(data, path):
    dataset = Dataset.from_dict({'text': data})
    dataset.save_to_disk(path)


def generate_dataset(dataset, path, word=False):
    # key is the file name, value is the list of tuples
    for key, value in dataset.items():
        print(f'Generating dataset: {path}/{key}')
        if word:
            word_ = []
        else:
            baseline_, spaced_, pn_ = [], [], []
        for triplet in value:
            a, b, operation = triplet
            # convert the string numbers to integers
            a, b = int(a), int(b)
            if word:
                p = inflect.engine()
                result = p.number_to_words(a + b)
                a, b = p.number_to_words(a), p.number_to_words(b)
                word_.append(f'Q: What is {a} plus {b}? A: {result}')
            else:
                baseline_.append(baseline(a, b, operation))
                spaced_.append(spaced(a, b, operation))
                pn_.append(pn(a, b, operation))
        if word:
            save_dataset(word_, f'{path}/{key}')
        else:
            save_dataset(baseline_, f'{path}/baseline/{key}')
            save_dataset(spaced_, f'{path}/spaced/{key}')
            save_dataset(pn_, f'{path}/positional/{key}')


def not_exceed(triplet):
    a, b, operation = triplet
    flag = True
    # check if the addition of either units, tens, etc. exceed 9
    length = max(len(a), len(b))
    if len(a) < length:
        a = '0' * (length - len(a)) + a
    if len(b) < length:
        b = '0' * (length - len(b)) + b
    for i in range(-1, -length-1, -1):
        if operation == '+':
            if int(a[i]) + int(b[i]) > 9:
                flag = False
                break
        elif operation == '-':
            if int(a[i]) - int(b[i]) < 0:
                flag = False
                break
        else:
            raise ValueError("Invalid type!")
    return flag


def main():
    parser = argparse.ArgumentParser(description="Load configuration variables from command-line arguments")
    parser.add_argument('--type', type=str, required=True, help='choose either `training` or `test`')
    parser.add_argument('--n-digits', nargs='+', type=int, default=[2, 3, 4, 5], help='list of integers')
    parser.add_argument('--n-samples', type=int, default=3000, help='number of samples to generate')
    parser.add_argument('--not-exceed', action='store_true', help='Not-exceed test set')
    parser.add_argument('--positive', action='store_true', help='Positive test set')
    parser.add_argument('--mul', action='store_true', help='choose only multiplication test set with at least one one-digit operand')
    parser.add_argument('--word', action='store_true', help='Use word numbers instead of digits')
    args = parser.parse_args()

    # Assign variables from parsed arguments
    dataset_type = args.type
    n_digits = args.n_digits
    n_samples = args.n_samples
    not_exceed = args.not_exceed
    positive = args.positive
    mul = args.mul
    word = args.word

    if not_exceed:
        dataset_type = 'not-exceed'
    if positive:
        dataset_type = 'positive'
    if mul:
        dataset_type = 'mul'
    if word:
        dataset_type = 'word'

    if dataset_type == 'training':
        # generate the pool
        pool = generate_pool(n_digits=n_digits, n_samples=n_samples)
        # generate the dataset from the pool
        path = 'datasets/training'
        generate_dataset(pool, path)
    elif dataset_type == 'test':
        # read the dataset
        dataset = read_dataset()
        # generate the dataset
        path = 'datasets/test'
        generate_dataset(dataset, path)
    elif dataset_type == 'not-exceed':
        # read the dataset
        dataset = read_dataset()
        # filter the dataset
        for key, value in dataset.items():
            if 'addition' in key or 'subtraction' in key:
                dataset[key] = [triplet for triplet in value if not_exceed(triplet)]
        # remove multiplication without showing the result
        dataset.pop('two_digit_multiplication')
        # generate the dataset
        path = 'datasets/test/not-exceed'
        generate_dataset(dataset, path)
    elif dataset_type == 'positive':
        # read the dataset
        dataset = read_dataset()
        # filter the dataset
        for key, value in dataset.items():
            if 'subtraction' in key:
                dataset[key] = [triplet for triplet in value if int(triplet[0]) - int(triplet[1]) >= 0]
        dataset.pop('two_digit_addition')
        dataset.pop('three_digit_addition')
        dataset.pop('four_digit_addition')
        dataset.pop('five_digit_addition')
        dataset.pop('six_digit_addition')
        dataset.pop('two_digit_multiplication')
        # generate the dataset
        path = 'datasets/test/positive'
        generate_dataset(dataset, path)
    elif dataset_type == 'mul':
        # read the dataset
        dataset = read_dataset()
        # filter the dataset
        for key, value in dataset.items():
            if 'multiplication' in key:
                dataset[key] = [triplet for triplet in value if len(triplet[0]) == 1 or len(triplet[1]) == 1]
        # generate the dataset
        path = 'datasets/test/mul'
        dataset.pop('two_digit_addition')
        dataset.pop('two_digit_subtraction')
        dataset.pop('three_digit_addition')
        dataset.pop('three_digit_subtraction')
        dataset.pop('four_digit_addition')
        dataset.pop('four_digit_subtraction')
        dataset.pop('five_digit_addition')
        dataset.pop('five_digit_subtraction')
        dataset.pop('six_digit_addition')
        dataset.pop('six_digit_subtraction')
        generate_dataset(dataset, path)
    elif dataset_type == 'word':
        # read the dataset
        dataset = read_dataset()
        # generate the dataset
        path = 'datasets/test/word'
        dataset.pop('two_digit_subtraction')
        dataset.pop('three_digit_addition')
        dataset.pop('three_digit_subtraction')
        dataset.pop('four_digit_addition')
        dataset.pop('four_digit_subtraction')
        dataset.pop('five_digit_addition')
        dataset.pop('five_digit_subtraction')
        dataset.pop('six_digit_addition')
        dataset.pop('six_digit_subtraction')
        dataset.pop('two_digit_multiplication')
        generate_dataset(dataset, path, word=True)
    else:
        raise ValueError("Invalid type!")


if __name__ == '__main__':
    main()
