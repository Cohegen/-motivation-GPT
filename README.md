# AI-Generated Motivational Quotes

Generate unique, deep-sounding motivational quotes using a fine-tuned GPT-2 model!

## Features
- Fine-tune GPT-2 on a dataset of famous motivational quotes
- Generate new quotes via a Python script or a web app
- Web app interface powered by Gradio

## Project Structure
-If you download the scripts in this repo, below is an authentic structure of how you should arrange these scripts in folder so that your work is easy in your IDE.
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

- ## Google Colab Usage

This repository includes a Jupyter notebook (`motivation_AI.ipynb`) designed for Google Colab. To use:
1. Open [motivation_AI.ipynb](./motivation_AI.ipynb) in Google Colab.
2. Upload your dataset (`quotes.json` or `quotes_plain.txt`) using the provided upload cell (see notebook instructions).
3. Run each cell sequentially to fine-tune the GPT-2 model and generate motivational quotes.
4. Launch the Gradio web interface from the notebook to interact with the model online.

## Example Output

> "Success is not final, failure is not fatal: it is the courage to continue that counts."  
> — AI-Generated-->Success is not final, failure is not fatal: it is the courage to continue that counts. And sometimes courage is not enough, it's the courage to be out of control and without direction.",
    "Author": "Albert Einstein",

## Dataset Format

- **quotes.json**: List of objects, each with `quote`, `author`, and optionally `tags`.
- **quotes_plain.txt**: One motivational quote per line.

## Training Parameters

- Epochs: 3
- Batch size: 2
- Block size: 64
- Learning rate: 5e-5

## Known Issues

- **GPU Requirement:** Model training requires a GPU; use Google Colab for free cloud GPUs.
- **Model Size:** Fine-tuned models may be too large to download or run locally; work directly in Colab when needed.
- **Transformers Dataset Warning:** You may see a warning about dataset deprecation; follow HuggingFace guidance for future upgrades.

## Quick Links

- [Kaggle - Quotes Dataset](https://www.kaggle.com/datasets/akmittal/quotes-dataset)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Gradio](https://gradio.app/)

---


---
Feel free to customize and expand this project! 
