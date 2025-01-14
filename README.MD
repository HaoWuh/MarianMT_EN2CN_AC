# MarianMT_EN2ZH_AC

This repository focuses on translating English to Chinese in the domain of aviation communication. Using a specialized dataset, we have trained a model capable of efficiently and accurately translating common and technical terms used in aviation communications.

## Introduction

This project provides a specialized translation model for the aviation communication field, addressing the unique challenges of translating technical terminology. By using the MarianMT framework, we ensure high efficiency as well as accuracy, making this tool valuable for professionals and applications in the aviation industry.

## Features

- **Domain-Specific Translation**: Specially trained for aviation communication, ensuring accurate translation of both common and technical terms.
- **High Accuracy**: Achieves high accuracy through rigorous training on a curated dataset of aviation communication texts.
- **Efficient Performance**: High efficiency due to the small size of the model, suitable for real-time translation needs.
- **Easy Integration**: The model can be easily integrated into existing systems using MarianMT's framework.

## Model Download and Usage

### Prerequisites

- Pytorch
- `transformers` library from Hugging Face

### usage

1. Clone this repository:
   ```bash
   git clone https://github.com/HaoWuh/MarianMT_EN2ZH_AC.git
   cd MarianMT_EN2ZH_AC
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Downloading the Model(checkpoints)

The trained model is shared on google drive, download link:
   ```
   https://drive.google.com/drive/folders/1DbzEcJxkBYVuOb2MACML14SWNZveAVuH?usp=drive_link
   ```
Or a download.sh is also provided in "./trained_model":

   ```bash
   ./download.sh
   ```

### Example Usage

To quickly test the model, you can run the demo script:
```bash
cd scripts
python Helsinki-en2zh_translate.py
```



