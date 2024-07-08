# Numerical Comprehension in LLMs

## Introduction
This project explores the capability of Large Language Models (LLMs), particularly BERT-based models, to understand and perform arithmetic tasks involving numerals.

## Datasets

## Fine-Tuning Parameters
- **Epochs**: 10
- **Batch size**: 32
- **Learning rate**: Started at 

<!-- ## Experimental Setup
### Reframing Numerals
This preprocessing strategy aims to improve numeral understanding by focusing on positional values, breaking down numerals into individual digits (e.g., "23" becomes "2 3").

### Custom Masking Mechanisms
A modified masking approach targets spans of consecutive digits, enhancing the model's ability to understand and predict entire numerals within arithmetic contexts. -->

## Installation
To set up the project, follow these steps:
1. Clone the repository:
```bash
git clone http://github.com/abbassix/numeracy
```
(optional) 1½. Create a virtual environment:
```bash
python3.10 -m venv .venv
```
1½.a. On Windows:
```bash
.venv\Scripts\activate
```
1½.b. On macOS/Linux:
```bash
source .venv/bin/activate
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

<!-- ## Usage
To run the project, you need to create the training and test datasets. But before that you need to have a directory named `datasets` in the parent directoy of the project to store the datasets there. You will also need to have two more directories: `models` to load to store models and `results` to store the resulting accuracies.
To make sure if you have the required directories and if not, to create them, run the following lines.
```bash
chmod +x create_dirs.sh
./create_dirs.sh
```
Then you need to run the following lines to create 6 different training and test datasets.
```bash
chmod +x create_datasets.sh
./create_datasets.sh
```
After creating datasets, you have to fine-tune the models. If the models are already stored in the `models` directory they will be loaded from there, otherwise they will be downloaded from Hugging Face Hub. The arguments to fine-tune the modela are stored in three different YAML files. Run the following lines to fine-tune three different scenarios to compare the results.
```bash
chmod +x finetune.sh
./finetune.sh
``` -->
