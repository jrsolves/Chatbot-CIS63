import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

# Load your dataset
df = pd.read_csv('static/science_projects.csv', sep=",")


# Suppose you want to train on the "Instruction" column
texts = df["Instruction"].tolist()

# Combine all instructions into one large text, or store them line-by-line
# For language modeling, you can concatenate them into a single file or keep them separate.
# Let's assume line-by-line:
dataset = Dataset.from_dict({"text": texts})

# Load tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Split into train and test
split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Load pre-trained GPT-2
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Training arguments
training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    logging_steps=100,
    eval_steps=500,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Save the model and tokenizer
model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")
