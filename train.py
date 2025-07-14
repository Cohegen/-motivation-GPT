import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

# Paths
DATA_PATH = os.path.join('data', 'quotes_plain.txt')
MODEL_OUTPUT_DIR = os.path.join('models', 'gpt2-finetuned')

# Hyperparameters
EPOCHS = 3
BATCH_SIZE = 2
BLOCK_SIZE = 64  # Max sequence length
LEARNING_RATE = 5e-5

# 1. Load GPT-2 tokenizer and model
print('Loading GPT-2 tokenizer and model...')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 2. Prepare dataset
def load_dataset(file_path, tokenizer, block_size=BLOCK_SIZE):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

dataset = load_dataset(DATA_PATH, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 3. Set up training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=LEARNING_RATE,
    logging_steps=100
)

# 4. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# 5. Train
print('Starting training...')
trainer.train()

# 6. Save the model
print(f'Saving model to {MODEL_OUTPUT_DIR}...')
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print('Training complete!') 