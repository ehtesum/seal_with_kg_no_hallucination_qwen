import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from transformers.utils import logging

# ==========================================
# CONFIG
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "../models/qwen_lora"
DATA_PATH = "../data/train.jsonl"

MAX_LENGTH = 256
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 2e-4
EPOCHS = 2

# Enable detailed logging
logging.set_verbosity_info()

# ==========================================
# 4-bit Quantization Config (QLoRA)
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ==========================================
# Load Tokenizer
# ==========================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# Load Model (4-bit)
# ==========================================
print("Loading model (4-bit QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

# ==========================================
# LoRA Configuration
# ==========================================
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# ==========================================
# Dataset Loader
# ==========================================
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def tokenize_function(example):
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ==========================================
# Training Arguments
# ==========================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,

    logging_dir="../logs",
    logging_steps=1,
    logging_strategy="steps",

    save_strategy="epoch",
    save_total_limit=2,

    fp16=True,
    optim="paged_adamw_8bit",

    report_to="mlflow",  # change to "tensorboard" if using TensorBoard
)

# ==========================================
# Trainer
# ==========================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# ==========================================
# Train
# ==========================================
if __name__ == "__main__":
    print("Starting training...\n")
    trainer.train()

    print("\nSaving LoRA model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nTraining complete.")