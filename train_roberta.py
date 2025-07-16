from transformers import RobertaTokenizer, RobertaForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load SQuAD 1.1
dataset = load_dataset("squad")

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForQuestionAnswering.from_pretrained("roberta-base")

# Preprocess function
def preprocess(example):
    inputs = tokenizer(
        example["question"], example["context"],
        truncation=True, padding="max_length", max_length=512
    )
    inputs["start_positions"] = example["answers"]["answer_start"][0]
    inputs["end_positions"] = example["answers"]["answer_start"][0] + len(example["answers"]["text"][0])
    return inputs

# Tokenize datasets
tokenized_dataset = dataset.map(preprocess, batched=False)

# Training args (200 epochs)
training_args = TrainingArguments(
    output_dir="./roberta-qa",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_total_limit=1,
    logging_steps=100,
    logging_dir="./logs",
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

# Fine-tune the model
trainer.train()
# Save model and tokenizer after training
model.save_pretrained("./models/squad11/roberta_qa_backdoored")
tokenizer.save_pretrained("./models/squad11/roberta-qa-backdoored")
