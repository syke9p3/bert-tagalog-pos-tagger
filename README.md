# BERT Tagalog Part of Speech Tagger (BERTTPOST)

### Cite this repository
Saya-ang, K., Hamor, M. G., Gozum, D. J., & Mabansag, R. K. Bidirectional Encoder Representation from Transformer Tagalog Part of Speech Tagger [Computer software]. https://github.com/syke9p3/bert-tagalog-pos-tagger

![https://github.com/syke9p3/bert-tagalog-pos-tagger/main/BERTTPOST%20Screenshot.jpg?raw=true](https://github.com/syke9p3/bert-tagalog-pos-tagger/blob/4fda52c3f9c00dfab6308fe725a2ae585304a081/BERTTPOST%20Screenshot.jpg)

This repository contains the training and testing Python files for fine-tuning [gklmip/bert-tagalog-base-uncased](https://huggingface.co/GKLMIP/bert-tagalog-base-uncased) model for Tagalog part of speech tagging 

- **Developed by:** Saya-ang, Kenth G. (@syke9p3) | Gozum, Denise Julianne S. (@Xenoxianne) | Hamor, Mary Grizelle D. (@mnemoria) | Mabansag, Ria Karen B. (@riavx)
- **Model type:** BERT Tagalog Base Uncased
- - **Programming Language:** Python
- **Languages (NLP):** Tagalog, Filipino
- **Dataset:** Sagum et. al.'s annotated Tagalog Corpora based on MGNN Tagset convention. This model was trained in 800 sentences and evaluated with 200 sentences.
- **Finetuned from model**: [Jiang et. al.'s pre-trained bert-tagalog-base-uncased model](https://huggingface.co/GKLMIP/bert-tagalog-base-uncased)

## HuggingFace Link
You can try the model in [HuggingFace Spaces](https://huggingface.co/spaces/syke9p3/bert-tagalog-base-uncased-pos-tagger?text=Naisip+ko+na+kumain+na+lang+tayo+sa+pinakasikat+na+restaurant+sa+Manila)

Model source code: [HuggingFace](https://huggingface.co/syke9p3/bert-tagalog-base-uncased-pos-tagger)

# Python Libraries
1. PyTorch
2. Regular Expressions
3. Transformers
4. SKLearn Metrics
5. Datasets
6. tqdm

# Dataset and Preprocessing
A corpus was used containing tagged sentences in Tagalog language. The dataset comprises sentences with each word annotated with its corresponding POS tag in the format of ```<TAG word>```. To prepare the corpus for training, the following preprocessing steps were performed:
1. **Removal of Line Identifier**: the line identifier, such as ```SNT.108970.2066```, was removed from each tagged sentence.
2. **Symbol Conversion**: for the BERT model, certain special symbols like hyphens, quotes, commas, etc., were converted into special tokens (```PMP```, ```PMS```, ```PMC```) to preserve their meaning during tokenization.
3. **Alignment of Tokenization**: the BERT tokenized words and their corresponding POS tags were aligned to ensure that the tokenization and tagging are consistent.


# Training

 The BERT Tagalog POS Tagger were trained using PyTorch library with the following hyperparameters set:

| **Hyperparamter**   |  **Value** |   
|---------------- |---------
| Batch Size      |  8 |
| Training Epoch  |  5 |
| Learning-rate   |  2e-5 |
| Optimizer       |  Adam |


# Inference

For the test sentences, almost the same preprocessing and tokenization steps as in training were performed, but without the need to extract POS tags from the sentence. The trained model was loaded to generate the tags for the input sentence along with [Gradio](https://www.gradio.app/docs/interface) to provide an interface for displaying the POS tag results.
