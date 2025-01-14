import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import sys
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import torch

# 加载预训练的翻译模型和分词器
# model_path = rf"D:\Work\Shanghai\FloatingAI\pretrained_MarianMT_model" 
# model_path = rf"C:\Work\VSCode\Python files\Shanghai\FloatingAI\EN2ZH\trained_model\model"
model_path = rf"C:\Work\VSCode\Python files\Shanghai\FloatingAI\EN2ZH\trained_model\model_1223"

model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型加载到 GPU（如果可用）
model = model.to(device)

# 输入的 DataFrame 示例
# data_path= rf"D:\Work\Shanghai\FloatingAI\大兴机场_100条\英文文语料列表1101_1130.csv"
data_path= rf"C:\Work\VSCode\Python files\Shanghai\FloatingAI\EN2ZH\test_datasets\en2zh_test_a.csv"
data= pd.read_csv(data_path)
try:
    src_text= list(data['text'])
except:
    try:
        src_text= list(data['en'])
    except:
        raise Exception("No column named 'text' or 'en' in the DataFrame")

# src_text= src_text[:1000]
# src_text= src_text[:100]

# 将单个文本编码为 Tensor
inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True)
# 使用模型生成翻译
inputs= inputs.to(device)

print("generating: ......")


# 翻译文本并保存到文件
output_file = "test_trained_a.csv"

translated_texts= []

for text in tqdm(src_text):
    # 将单个文本编码为 Tensor，并移动到 GPU
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 确保所有张量在同一设备上
    # 使用模型生成翻译
    outputs = model.generate(**inputs)
    # 将生成的结果解码为字符串
    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 写入文件
    translated_texts.append(translated)

save_df= pd.DataFrame({
    "en": src_text,
    "zh": translated_texts
})
save_df.to_csv(output_file, index=False)

print(f"Translation saved to {output_file}")
