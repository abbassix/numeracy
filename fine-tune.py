# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:00:00 2024

Author: Mehdi Abbasi
GitHub: abbassix
"""

# import the necessary libraries
import sys
import os
import argparse
import warnings
import logging
import json
from functions import load_model, add_tokens, load_dataset, fine_tune, accuracy
from inputimeout import inputimeout, TimeoutOccurred


logging.basicConfig(level=logging.WARNING)

print("Successfully imported the necessary libraries.\n")


def main():
    parser = argparse.ArgumentParser(description="Load configuration variables from command-line arguments")
    parser.add_argument('--model-checkpoint-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--approach', type=str, required=True, help='Approach to train the model: `baseline`, `spaced` or `positional`')
    parser.add_argument('--operation', type=str, required=True, help='op to train the model: `addition`, `subtraction` or `multiplication`')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs for training')
    parser.add_argument("--learning-rate", nargs='+', type=float, help="Learning rates for training")
    parser.add_argument('--data-collator', type=str, required=True, help='Data collator to use: `default` or `custom`')
    parser.add_argument('--new-token', nargs='+', type=str, required=False, help='New token to be added')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--n-sample', type=int, required=False, default=100, help='Number of samples to take from the test set')

    args = parser.parse_args()

    # Assign variables from parsed arguments
    model_checkpoint_path = args.model_checkpoint_path
    approach = args.approach
    op = args.operation
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    data_collator = args.data_collator
    new_token = args.new_token
    verbose = args.verbose
    n_sample = args.n_sample

    # For demonstration, print out the variables
    print("Model Checkpoint Path:", model_checkpoint_path)
    print("Approach:", approach)
    print("op:", op)
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)
    print("Learning Rate:", learning_rate)
    print("Data Collator:", data_collator)
    print("New Token:", new_token)
    print("Verbose:", verbose)

    model_name = model_checkpoint_path.split("/")[-1]
    train_ds_path = f'datasets/training/{approach}/{op}'

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Loading the model and the train dataset...\n")
    try:
        model, tokenizer = load_model(model_checkpoint_path)
    except FileNotFoundError as e:
        warnings.warn(f"Error: Model '{model_checkpoint_path}' doesn't exist.")
        logging.error(f"Encountered a problem: {str(e)}")
        sys.exit(1)

    try:
        train_dataset = load_dataset(train_ds_path)
    except FileNotFoundError as e:
        warnings.warn(f"Error: train dataset '{train_ds_path}' doesn't exist.")
        logging.error(f"Encountered a problem: {str(e)}")
        sys.exit(1)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # add the new token to the model's vocabulary
    if new_token:
        print("Adding a new token to the model...\n")
        add_tokens(model, tokenizer, new_token)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Tokenizing the train dataset...\n")

    def tokenize_label(example: dict) -> dict:
        return tokenizer(example["text"])

    tokenized_dataset = train_dataset.map(
        tokenize_label,
        batched=True,
        remove_columns=["text"])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print(f"Checking initial accuracies on {n_sample} samples...\n")

    def show_accuracies(accs):
        for key, lst in dict.items(accs):
            print(f"Accuracy on {key} test set: ", end='')
            if len(lst) == 1:
                print(f"{lst[0]}%")
                continue
            for i in range(len(lst) - 1):
                print(lst[i], end='% -> ')
            if lst[-1] > lst[-2]:
                print(f"{lst[-1]}% (↑{lst[-1] - lst[-2]}%)")
            elif lst[-1] == lst[-2]:
                print(f"{lst[-1]}% (-)")
            else:
                print(f"{lst[-1]}% (↓{lst[-2] - lst[-1]}%)")

    if op == "addition" or op == "subtraction":
        accuracies = {'two_digit': [], 'three_digit': [], 'four_digit': [], 'five_digit': [], 'six_digit': []}
    elif op == "multiplication":
        accuracies = {'two_digit': []}

    if op == "addition" or op == "subtraction":
        _2d_test = load_dataset(f"datasets/test/{approach}/two_digit_{op}")
        # sample the test set to speed up the process
        _2d_test = _2d_test.shuffle(seed=42).select(range(n_sample))
        two_digit_accuracy = accuracy(model, tokenizer, _2d_test)
        accuracies['two_digit'].append(two_digit_accuracy)
        # show_accuracies(accuracies['two_digit']) if verbose else None
        _3d_test = load_dataset(f"datasets/test/{approach}/three_digit_{op}")
        # sample the test set to speed up the process
        _3d_test = _3d_test.shuffle(seed=42).select(range(n_sample))
        three_digit_accuracy = accuracy(model, tokenizer, _3d_test)
        accuracies['three_digit'].append(three_digit_accuracy)
        # show_accuracies(accuracies['three_digit']) if verbose else None
        _4d_test = load_dataset(f"datasets/test/{approach}/four_digit_{op}")
        # sample the test set to speed up the process
        _4d_test = _4d_test.shuffle(seed=42).select(range(n_sample))
        four_digit_accuracy = accuracy(model, tokenizer, _4d_test)
        accuracies['four_digit'].append(four_digit_accuracy)
        # show_accuracies(accuracies['four_digit']) if verbose else None
        _5d_test = load_dataset(f"datasets/test/{approach}/five_digit_{op}")
        # sample the test set to speed up the process
        _5d_test = _5d_test.shuffle(seed=42).select(range(n_sample))
        five_digit_accuracy = accuracy(model, tokenizer, _5d_test)
        accuracies['five_digit'].append(five_digit_accuracy)
        # show_accuracies(accuracies['five_digit']) if verbose else None
        _6d_test = load_dataset(f"datasets/test/{approach}/six_digit_{op}")
        # sample the test set to speed up the process
        _6d_test = _6d_test.shuffle(seed=42).select(range(n_sample))
        six_digit_accuracy = accuracy(model, tokenizer, _6d_test)
        accuracies['six_digit'].append(six_digit_accuracy)
        # show_accuracies(accuracies['six_digit']) if verbose else None
        show_accuracies(accuracies) if verbose else None
    elif op == "multiplication":
        _2d_test = load_dataset(f"datasets/test/{approach}/two_digit_{op}")
        # sample the test set to speed up the process
        _2d_test = _2d_test.shuffle(seed=42).select(range(n_sample))
        two_digit_accuracy = accuracy(model, tokenizer, _2d_test)
        accuracies['two_digit'].append(two_digit_accuracy)
        # show_accuracies(accuracies['two_digit']) if verbose else None
        show_accuracies(accuracies) if verbose else None
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Fine-tuning the model...\n")
    for i in range(epochs):
        print(f"epoch {i+1} of {epochs} epochs.")
        # in case the number of epochs is more than the number of
        # learning rates provided use the last learning rate
        if i >= len(learning_rate):
            # warnings.warn("Warning: The number of epochs is more than the number of learning rates provided.")
            logging.warning("Using the last learning rate for the rest of the epochs.")
            lr = learning_rate[-1]
        # otherwise, use the corresponding learning rate
        else:
            lr = learning_rate[i]
        # shuffle the dataset
        tokenized_dataset = tokenized_dataset.shuffle(seed=42)
        model = fine_tune(model,
                          tokenizer,
                          tokenized_dataset,
                          batch_size=batch_size,
                          num_epochs=1,
                          lr=lr,
                          collator=data_collator,
                          weight_decay=0.0)
        # test the accuracy of the model after each epoch
        print("Checking accuracies...\n")
        if op == "addition" or op == "subtraction":
            two_digit_accuracy = accuracy(model, tokenizer, _2d_test)
            accuracies['two_digit'].append(two_digit_accuracy)
            # show_accuracies(accuracies['two_digit']) if verbose else None
            three_digit_accuracy = accuracy(model, tokenizer, _3d_test)
            accuracies['three_digit'].append(three_digit_accuracy)
            # show_accuracies(accuracies['three_digit']) if verbose else None
            four_digit_accuracy = accuracy(model, tokenizer, _4d_test)
            accuracies['four_digit'].append(four_digit_accuracy)
            # show_accuracies(accuracies['four_digit']) if verbose else None
            five_digit_accuracy = accuracy(model, tokenizer, _5d_test)
            accuracies['five_digit'].append(five_digit_accuracy)
            # show_accuracies(accuracies['five_digit']) if verbose else None
            six_digit_accuracy = accuracy(model, tokenizer, _6d_test)
            accuracies['six_digit'].append(six_digit_accuracy)
            # show_accuracies(accuracies['six_digit']) if verbose else None
            show_accuracies(accuracies) if verbose else None
        elif op == "multiplication":
            two_digit_accuracy = accuracy(model, tokenizer, _2d_test)
            accuracies['two_digit'].append(two_digit_accuracy)
            # show_accuracies(accuracies['two_digit']) if verbose else None
            show_accuracies(accuracies) if verbose else None
        # ask user if she wants to continue training
        # if so (`y` or `yes`), continue training
        # otherwise (`n` or `no`), break the loop
        # is the user doesn't provide any input in 10 seconds, continue training
        # if it is the last epoch, break the loop
        if i == epochs - 1:
            print("Fine-tuning completed.\n")
            break
        try:
            # beep to notify the user
            os.system("printf '\a'")
            os.system("printf '\a'")
            # wait for 1.5 seconds
            os.system("sleep 1.5")
            os.system("printf '\a'")
            user_input = inputimeout(prompt="Do you want to continue training? (y/n/s): ", timeout=10)
            if user_input.lower() in ['n', 'no']:
                print("Stopping training.")
                break
            elif user_input.lower() in ['y', 'yes']:
                print("Continuing training.")
            elif user_input.lower() in ['s', 'save']:
                print("Saving the model and the tokenizer and the continuing training...\n")
                # save the model and the tokenizer
                saving_path = f"../models/{model_name}_{approach}_{op}_{data_collator}_epoch_{i+1}"
                model.save_pretrained(saving_path)
                tokenizer.save_pretrained(saving_path)
            else:
                print("Invalid input. Continuing training by default.")
        except TimeoutOccurred:
            print("No input received in 10 seconds. Continuing...")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Checking final accuracies...\n")
    if op == "addition" or op == "subtraction":
        two_digit_test_set = load_dataset(f"datasets/test/{approach}/two_digit_{op}")
        two_digit_accuracy = accuracy(model, tokenizer, two_digit_test_set)
        accuracies['two_digit'].append(two_digit_accuracy)
        print(f"Final accuracy on two-digit {op} test set: {two_digit_accuracy}") if verbose else None
        three_digit_test_set = load_dataset(f"datasets/test/{approach}/three_digit_{op}")
        three_digit_accuracy = accuracy(model, tokenizer, three_digit_test_set)
        accuracies['three_digit'].append(three_digit_accuracy)
        print(f"Final accuracy on three-digit {op} test set: {three_digit_accuracy}") if verbose else None
        four_digit_test_set = load_dataset(f"datasets/test/{approach}/four_digit_{op}")
        four_digit_accuracy = accuracy(model, tokenizer, four_digit_test_set)
        accuracies['four_digit'].append(four_digit_accuracy)
        print(f"Final accuracy on four-digit {op} test set: {four_digit_accuracy}") if verbose else None
        five_digit_test_set = load_dataset(f"datasets/test/{approach}/five_digit_{op}")
        five_digit_accuracy = accuracy(model, tokenizer, five_digit_test_set)
        accuracies['five_digit'].append(five_digit_accuracy)
        print(f"Final accuracy on five-digit {op} test set: {five_digit_accuracy}") if verbose else None
        six_digit_test_set = load_dataset(f"datasets/test/{approach}/six_digit_{op}")
        six_digit_accuracy = accuracy(model, tokenizer, six_digit_test_set)
        accuracies['six_digit'].append(six_digit_accuracy)
        print(f"Final accuracy on six-digit {op} test set: {six_digit_accuracy}") if verbose else None
    elif op == "multiplication":
        two_digit_test_set = load_dataset(f"datasets/test/{approach}/two_digit_{op}")
        two_digit_accuracy = accuracy(model, tokenizer, two_digit_test_set)
        accuracies['two_digit'].append(two_digit_accuracy)
        print(f"Final accuracy on two-digit {op} test set: {two_digit_accuracy}") if verbose else None
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("Saving the model and the tokenizer...\n")
    # save the model and the tokenizer
    saving_path = f"../models/{model_name}_{approach}_{op}_{data_collator}"
    model.save_pretrained(saving_path)
    tokenizer.save_pretrained(saving_path)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # save the accuracy results in a JSON file
    print("Saving the accuracy results...\n")
    results_file = f"results/{model_name}_{approach}_{op}_{data_collator}.json"
    # create a file to save the results
    with open(results_file, "w") as f:
        json.dump(accuracies, f)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


if __name__ == "__main__":
    main()
