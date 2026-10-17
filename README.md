# Webpage to Markdown with ReaderLM-v2

A small Streamlit app that fetches a webpage and uses the locally downloaded Jina ReaderLM-v2 model to convert its HTML into Markdown. CUDA and Apple MPS are selected automatically when available.

## Setup

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your machine if needed. On Apple Silicon, install the macOS PyTorch build to use MPS; on NVIDIA systems, install a CUDA-enabled PyTorch build matching your CUDA setup. The app falls back to CPU when neither accelerator is available.

`torchvision` is included because Transformers imports image-processing components while initializing model support. Keep its version aligned with the installed PyTorch version. If you already have a virtual environment, install the updated requirements into that environment:

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Enter a URL (the default is `https://www.elastic.co/`) and select **Convert to Markdown**. On the first conversion, Transformers downloads `jinaai/ReaderLM-v2` from Hugging Face into its normal model cache. The default output limit can be adjusted in the UI.

ReaderLM-v2 is licensed under [CC BY-NC 4.0](https://huggingface.co/jinaai/ReaderLM-v2). Review that license before commercial use.
