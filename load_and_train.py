from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import argparse

def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset

def train_dataset(train_file):
    # Configurations
    model_name = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    config = GPT2Config.from_pretrained(model_name)

    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    # Load dataset
    train_dataset = load_dataset(train_file, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,
    )
    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./output")
    tokenizer.save_pretrained("./output")
    
if __name__ == "__main__":
    # create an arguement for the file path using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", type=str, default='linkedin_posts.txt', help="Path to the training file")
    args = parser.parse_args()
    train_file_path = args.train_file_path
    train_dataset(train_file_path)
    