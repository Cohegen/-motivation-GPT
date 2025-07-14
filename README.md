# AI-Generated Motivational Quotes

Generate unique, deep-sounding motivational quotes using a fine-tuned GPT-2 model!

## Features
- Fine-tune GPT-2 on a dataset of famous motivational quotes
- Generate new quotes via a Python script or a web app
- Web app interface powered by Gradio

## Project Structure
```
motivationAI/
│
├── data/                # Dataset files (e.g., quotes.json, quotes_plain.txt)
├── models/              # Fine-tuned model files (e.g., gpt2-finetuned/)
├── scripts/             # Training and generation scripts
│   ├── train.py         # Fine-tune GPT-2
│   └── generate.py      # Generate quotes
├── app/                 # Web app code
│   └── app.py           # Gradio web app
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup
1. **Clone the repository and navigate to the project folder.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your dataset:**
   - Place your motivational quotes dataset in `data/quotes.json`.
   - Convert it to plain text (one quote per line) as `data/quotes_plain.txt`.
4. **Fine-tune the model:**
   - (Optional) Use `scripts/train.py` to fine-tune GPT-2 on your dataset.
   - Place your fine-tuned model in `models/gpt2-finetuned/`.

## Usage
### Generate Quotes via Script
```bash
python scripts/generate.py "Your prompt here"
```
- Leave the prompt blank for a random quote.

### Run the Web App
```bash
python app/app.py
```
- This will launch a Gradio web interface in your browser.

## Using Google Colab
- You can fine-tune and use the model in Google Colab if you don't have a local GPU.
- Upload your model folder to Colab and use the provided scripts or notebook.

## Limitations & Workarounds
- **Local Training Limitations:** Training GPT-2 locally was not possible due to hardware (GPU/VRAM) constraints.
- **Cloud Training:** I used Google Colab to fine-tune the model using a cloud-based GPU.
- **Model Size Issues:** The fine-tuned model was too large to download and run locally.
- **Cloud Deployment:** As a workaround, both quote generation and web app deployment were performed directly on Google Colab, using Gradio to provide a public web interface.

## Dataset
- **Source:** [Kaggle - Quotes Dataset](https://www.kaggle.com/datasets/akmittal/quotes-dataset)
- **File:** `quotes.json` (downloaded from Kaggle)
- The dataset contains thousands of motivational and famous quotes with authors and tags.

## Credits
- Built with [HuggingFace Transformers](https://huggingface.co/transformers/), [PyTorch](https://pytorch.org/), and [Gradio](https://gradio.app/).
- Motivational quotes dataset: [Kaggle - Quotes Dataset](https://www.kaggle.com/datasets/akmittal/quotes-dataset)

---
Feel free to customize and expand this project! 