# Import libraries

import pandas as pd
from transformers import AutoTokenizer

# Load clean_text.csv

df = pd.read_csv("./dataset/clean_text.csv", encoding="latin1")

print(df.head())

texts = df["clean_text"].dropna().tolist()

print("Number of texts:", len(texts))

# Select the first text

text = texts[0]

print("Original:", text)

# Load BERT tokenizer

bert = AutoTokenizer.from_pretrained("bert-base-uncased")

print("BERT tokenizer loaded")

# BERT -> Tokens

bert_tokens = bert.tokenize(text)

print("BERT TOKENS:")
print(bert_tokens)

# BERT -> Token IDs

bert_ids = bert.encode(text, add_special_tokens=True)

print("BERT TOKEN IDS:")
print(bert_ids)

# BERT: Token IDs -> Text

bert_decode = bert.decode(bert_ids)

print("BERT DECODED TEXT:")
print(bert_decode)

# Load GPT-2 Tokenizer

gpt2 = AutoTokenizer.from_pretrained("gpt2")

print("GPT-2 tokenizer loaded")

# GPT-2: Text -> Tokens

gpt2_tokens = gpt2.tokenize(text)

print("GPT-2 TOKENS:")
print(gpt2_tokens)

# GPT-2: Text -> Token IDs

gpt2_ids = gpt2.encode(text, add_special_tokens=False)

print("GPT-2 TOKEN IDS:")
print(gpt2_ids)

# GPT-2: Token IDs -> Text

gpt2_decode = gpt2.decode(gpt2_ids)

print("GPT-2 DECODED TEXT:")
print(gpt2_decode)

# Compare BERT and GPT-2

print("Original:", text)

print("\nBERT TOKENS:")
print(bert_tokens)

print("\nGPT-2 TOKENS:")
print(gpt2_tokens)

print("\nBERT TOKEN COUNT:", len(bert_tokens))

print("GPT-2 TOKEN COUNT:", len(gpt2_tokens))

# Process All Texts

results = []

for text in texts:
    b_tokens = bert.tokenize(text)
    g_tokens = gpt2.tokenize(text)

    results.append({
        "Original Text": text,
        "BERT Tokens": " ".join(b_tokens),
        "GPT-2 Tokens": " ".join(g_tokens),
        "BERT Count": len(b_tokens),
        "GPT-2 Count": len(g_tokens)
    })

comparison = pd.DataFrame(results)

print(comparison)

# Save comparison.csv

comparison.to_csv("./dataset/comparison.csv", index=False)

print("comparison.csv created")

# Create token_ids.csv

token_rows = []

for text in texts:
    b_tokens = bert.tokenize(text)
    b_ids = bert.convert_tokens_to_ids(b_tokens)

    g_tokens = gpt2.tokenize(text)
    g_ids = gpt2.convert_tokens_to_ids(g_tokens)

    token_rows.append({
        "Original Text": text,
        "BERT Tokens": " ".join(b_tokens),
        "BERT Token IDs": str(b_ids),
        "GPT-2 Tokens": " ".join(g_tokens),
        "GPT-2 Token IDs": str(g_ids)
    })

token_ids = pd.DataFrame(token_rows)

token_ids.to_csv("./dataset/token_ids.csv", index=False)

print("token_ids.csv created")

# Create vocabulary.txt

bert_vocab = bert.get_vocab()
gpt2_vocab = gpt2.get_vocab()

with open("./dataset/vocabulary.txt", "w", encoding="utf-8") as f:

    f.write("===== BERT VOCABULARY =====\n")

    for token, token_id in bert_vocab.items():
        f.write(f"{token} : {token_id}\n")

    f.write("\n===== GPT-2 VOCABULARY =====\n")

    for token, token_id in gpt2_vocab.items():
        f.write(f"{token} : {token_id}\n")

print("vocabulary.txt created")

# Check vocabulary sizes

print("BERT vocabulary size:", bert.vocab_size)

print("GPT-2 vocabulary size:", gpt2.vocab_size)

print("BERT tokens in dataset:",
      comparison["BERT Count"].sum())

print("GPT-2 tokens in dataset:",
      comparison["GPT-2 Count"].sum())

# Find highest token count

print("Highest BERT count:",
      comparison["BERT Count"].max())

print("Highest GPT-2 count:",
      comparison["GPT-2 Count"].max())

