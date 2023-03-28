import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

def finish_sentence(model, tokenizer, prompt, max_length=1000, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Create attention mask and pad token id
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id
    
    output_sequences = model.generate(
        input_ids=input_ids,
        min_length=100,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=3,  # Increase n-gram size to avoid repetition of phrases
        top_k=50,  # Set the top_k parameter to focus on more relevant tokens
        top_p=0.95,  # Set the top_p parameter to limit the token choice
        temperature=0.7,
        attention_mask=attention_mask,  # Set attention mask
        pad_token_id=pad_token_id,  # Set pad token id
    )

    generated_texts = []

    for generated_sequence in output_sequences:
        generated_sequence = generated_sequence.tolist()
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        generated_texts.append(text)

    return generated_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-text", type=str, default='', help="input text string")
    args = parser.parse_args()
    model_path = "./output"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    generated_texts = finish_sentence(model, tokenizer, args.input_text)
    print(generated_texts)