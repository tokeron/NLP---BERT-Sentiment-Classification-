# imports
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, \
    BertForMaskedLM, IntervalStrategy, DataCollatorForLanguageModeling
from sklearn.metrics import f1_score
from datasets import load_dataset, load_metric
import transformers

# path to the data
baby_path = Path("clean_data/baby")
office_products_path = Path("clean_data/baby")
OUT_PATH = "output"


# Function that alculates f1
def metric_fn(predictions):
    preds = predictions.predictions.argmax(axis=1)
    labels = predictions.label_ids
    return {'f1': f1_score(preds, labels, average='binary')}


if __name__ == '__main__':
    # Define model and get tokenizer and models from the best library in the world - huggingface
    model_name = 'bert-base-uncased'
    bert_for_pretraining = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model_seq_classification = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Defining The Datasets
    unlabeled_data_files = {
        'baby': str(baby_path / 'unlabeled.csv'),
        'office_products': str(office_products_path / 'unlabeled.csv')
    }
    data_files = {
        'train': str(baby_path / 'train.csv'),
        'test': str(office_products_path / 'dev.csv')
    }

    # Load data
    unlabeled_raw_datasets = load_dataset("csv", data_files=unlabeled_data_files)
    raw_datasets = load_dataset("csv", data_files=data_files)

    # Tokenize the data
    tokenized_unlabeled_datasets = unlabeled_raw_datasets.map(tokenizer, input_columns='review',
                                                              fn_kwargs={"max_length": 256, "truncation": True,
                                                                         "padding": "max_length"})

    tokenized_datasets = raw_datasets.map(tokenizer, input_columns='review',
                                          fn_kwargs={"max_length": 256, "truncation": True, "padding": "max_length"})

    tokenized_unlabeled_datasets.set_format('torch')
    tokenized_datasets.set_format('torch')

    # Unsupervised training by unlabeled data
    # define the arguments for the trainer
    bert_pretraining_args = TrainingArguments(
        output_dir='pytorch_finetuned_model',  # output directory
        num_train_epochs=3,  # total # of training epochs
        per_device_train_batch_size=8,  # batch size per device during training (try 16 if needed)
        per_device_eval_batch_size=8,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='pytorch_finetuned_log',  # directory for storing logs
        do_train=True,
        # evaluation_strategy=IntervalStrategy("steps"),
        eval_steps=2
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    bert_baby_pretrainer = Trainer(
        model=bert_for_pretraining,
        args=bert_pretraining_args,
        train_dataset=tokenized_unlabeled_datasets['baby'],
        eval_dataset=tokenized_unlabeled_datasets['baby']
    )

    # bert_office_products_pretrainer = Trainer(
    #     model=bert_for_pretraining,
    #     args=bert_pretraining_args,
    #     train_dataset=tokenized_unlabeled_datasets['office_products'],
    #     eval_dataset=tokenized_unlabeled_datasets['office_products']
    # )

    # Pretraining The Model
    bert_baby_pretrainer.train()
    # bert_office_products_pretrainer.train()


    # Prepare labels for the supervized learning
    for split in tokenized_datasets:
        tokenized_datasets[split] = tokenized_datasets[split].add_column('label', raw_datasets[split]['label'])

    args = TrainingArguments(output_dir=OUT_PATH, overwrite_output_dir=True, per_device_train_batch_size=16,
                             per_device_eval_batch_size=16, metric_for_best_model='dev_f1',
                             greater_is_better=True, do_train=True,
                             num_train_epochs=8,  evaluation_strategy="epoch")

    trainer = Trainer(
        model=model_seq_classification,
        args=args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=metric_fn)

    # Training The Model
    trainer.train()

    print(trainer.evaluate())



