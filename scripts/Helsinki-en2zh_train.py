import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import pandas as pd
import pyarrow as pa
from datasets import load_dataset, Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datetime import datetime

# 1. load the datasets
# read an load the datasets for training, validation and testing
train_dataset = load_dataset('csv', data_files={'train': '../datasets/train.tsv'}, delimiter='\t')
val_dataset = load_dataset('csv', data_files={'validation': '../datasets/val.tsv'}, delimiter='\t')
test_dataset = load_dataset('csv', data_files={'test': '../datasets/test.tsv'}, delimiter='\t')

train_dataset= train_dataset['train']
val_dataset= val_dataset['validation']
test_dataset= test_dataset['test']

# 2. load model and tokenizer
model_path = "/pretrained_MarianMT_model"  # can be downloaded from huggingface official websites
tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# checking available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)


# define tokenizer function
def tokenize_function(examples):
    # tokenize the en and zh sentences
    inputs= tokenizer(examples['en'], text_target=examples['zh'], padding="max_length", truncation=True)

    inputs=inputs.to(device)

    return inputs

# tokenization of training, validating and testing datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)


# arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',          # save path
    evaluation_strategy="epoch",     # evaluation frequency
    learning_rate=2e-5,             # learning rate
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=8,   # batch size for evaluation
    num_train_epochs=3,             # number of epochs
    logging_dir='./logs',           # logging path
    logging_steps=10,               # logging frequency
    save_steps=500,                 # saving frequency
    save_total_limit=2,             # ckeckpoint number limit
)

# create Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# training
trainer.train()

# saving
today = datetime.now()
date_string = today.strftime("%Y%m%d")
model_repo= 'model_'+date_string
model.save_pretrained('../trained_model/'+model_repo)
tokenizer.save_pretrained('../trained_model/'+model_repo)

print("Model saved! ", model_repo)
