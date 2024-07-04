# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:00:00 2024

Author: Mehdi Abbasi
GitHub: abbassix
"""

# import the necessary libraries
import sys
import argparse
import warnings
import logging
from functions import load_model, load_dataset, accuracy


logging.basicConfig(level=logging.WARNING)

print("Successfully imported the necessary libraries.\n")


def main():
    parser = argparse.ArgumentParser(description="Load configuration variables from command-line arguments")
    parser.add_argument('--approach', type=str, required=True, help='Approach to train the model: `baseline`, `spaced` or `positional`')
    parser.add_argument('--operation', type=str, required=True, help='op to train the model: `addition`, `subtraction` or `multiplication`')
    parser.add_argument('--not-exceed', action='store_true', help='Not-exceed test set')
    parser.add_argument('--positive', action='store_true', help='Positive test set')
    parser.add_argument('--mul', action='store_true', help='divide multiplication test set into two-digit and three-digit numbers')
    parser.add_argument('--word', action='store_true', help='Use word numbers instead of digits')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--verbose-positive', action='store_true', help='Print verbose output for positive test set')
    parser.add_argument('--verbose-negative', action='store_true', help='Print verbose output for negative test set')

    args = parser.parse_args()

    # Assign variables from parsed arguments
    approach = args.approach
    op = args.operation
    not_exceed = args.not_exceed
    positive = args.positive
    mul = args.mul
    word = args.word
    verbose = args.verbose
    verbose_positive = args.verbose_positive
    verbose_negative = args.verbose_negative

    model_checkpoint_path = f'distilbert/distilbert-base-uncased_{approach}_{op}_custom'
    if not_exceed:
        test_case = 'not-exceed/'
    elif positive:
        test_case = 'positive/'
    elif mul:
        test_case = 'mul/'
    elif word:
        test_case = 'word'
    else:
        test_case = ''
    
    if verbose_positive:
        v = 'positive'
    elif verbose_negative:
        v = 'negative'
    else:
        v = ''

    # For demonstration, print out the variables
    print("Approach:", approach)
    print("op:", op)
    print("Verbose:", verbose)

    model_name = model_checkpoint_path.split("/")[-1]

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Loading the model...\n")
    try:
        model, tokenizer = load_model(model_checkpoint_path)
        print(f"Model '{model_name}' is loaded.") if verbose else None
    except FileNotFoundError as e:
        warnings.warn(f"Error: Model '{model_checkpoint_path}' doesn't exist.")
        logging.error(f"Encountered a problem: {str(e)}")
        sys.exit(1)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if op == "addition" or op == "subtraction":
        accuracies = {'two_digit': [], 'three_digit': [], 'four_digit': [], 'five_digit': [], 'six_digit': []}
    elif op == "multiplication":
        accuracies = {'two_digit': []}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Checking final accuracies...\n")
    if op == "addition" or op == "subtraction":
        if word:
            two_digit_test_set = load_dataset(f"datasets/test/{test_case}/two_digit_{op}")
            two_digit_accuracy = accuracy(model, tokenizer, two_digit_test_set, verbose=v)
            print(f"Final accuracy on two-digit word {op} test set: {two_digit_accuracy}") if verbose else None
        else:
            two_digit_test_set = load_dataset(f"datasets/test/{test_case}{approach}/two_digit_{op}")
            two_digit_accuracy = accuracy(model, tokenizer, two_digit_test_set, verbose=v)
            accuracies['two_digit'].append(two_digit_accuracy)
            print(f"Final accuracy on two-digit {op} test set: {two_digit_accuracy}") if verbose else None
            three_digit_test_set = load_dataset(f"datasets/test/{test_case}{approach}/three_digit_{op}")
            three_digit_accuracy = accuracy(model, tokenizer, three_digit_test_set, verbose=v)
            accuracies['three_digit'].append(three_digit_accuracy)
            print(f"Final accuracy on three-digit {op} test set: {three_digit_accuracy}") if verbose else None
            four_digit_test_set = load_dataset(f"datasets/test/{test_case}{approach}/four_digit_{op}")
            four_digit_accuracy = accuracy(model, tokenizer, four_digit_test_set, verbose=v)
            accuracies['four_digit'].append(four_digit_accuracy)
            print(f"Final accuracy on four-digit {op} test set: {four_digit_accuracy}") if verbose else None
            five_digit_test_set = load_dataset(f"datasets/test/{test_case}{approach}/five_digit_{op}")
            five_digit_accuracy = accuracy(model, tokenizer, five_digit_test_set, verbose=v)
            accuracies['five_digit'].append(five_digit_accuracy)
            print(f"Final accuracy on five-digit {op} test set: {five_digit_accuracy}") if verbose else None
            six_digit_test_set = load_dataset(f"datasets/test/{test_case}{approach}/six_digit_{op}")
            six_digit_accuracy = accuracy(model, tokenizer, six_digit_test_set, verbose=v)
            accuracies['six_digit'].append(six_digit_accuracy)
            print(f"Final accuracy on six-digit {op} test set: {six_digit_accuracy}") if verbose else None
    elif op == "multiplication":
        two_digit_test_set = load_dataset(f"datasets/test/{test_case}{approach}/two_digit_{op}")
        two_digit_accuracy = accuracy(model, tokenizer, two_digit_test_set, verbose=v)
        accuracies['two_digit'].append(two_digit_accuracy)
        print(f"Final accuracy on two-digit {op} test set: {two_digit_accuracy}") if verbose else None
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == "__main__":
    main()
