import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import torch

# loading the trained model and tokenizer
model_path = "../trained_model/model_1223"
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# checking available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)

# reading demo data
# This a demo data with both en and zh columns as original texts and labels repectively
data_path= "../demo_data/en2zh_test.csv"
data= pd.read_csv(data_path)
try:
    src_text= list(data['text'])
except:
    try:
        src_text= list(data['en'])
    except:
        raise Exception("No column named 'text' or 'en' in the DataFrame")


# tokenize input texts
inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
inputs= inputs.to(device)

print("generating: ......")

output_file = "../results/test_trained.csv"

translated_texts= []

for text in tqdm(src_text):
    # moving the tokenized tensors into GPU device
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()} 
    # translating
    outputs = model.generate(**inputs)
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # writing
    translated_texts.append(translated)

save_df= pd.DataFrame({
    "en": src_text,
    "zh": translated_texts
})
save_df.to_csv(output_file, index=False)

print(f"Translation saved to {output_file}")
