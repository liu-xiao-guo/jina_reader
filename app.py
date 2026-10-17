"""Convert a webpage to Markdown with the local Jina ReaderLM-v2 model."""

from __future__ import annotations

import re
from urllib.parse import urlparse

import requests
import streamlit as st
import torch
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "jinaai/ReaderLM-v2"
MAX_HTML_CHARS = 300_000


def get_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA GPU"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Metal (MPS)"
    return torch.device("cpu"), "CPU"


def fetch_html(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a complete http:// or https:// URL.")
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; ReaderLM-Streamlit/1.0)"},
        timeout=(10, 45),
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if content_type and "html" not in content_type.lower():
        raise ValueError(f"The URL returned {content_type}, not an HTML page.")
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html[:MAX_HTML_CHARS], "html.parser")
    for node in soup(["script", "style", "noscript", "svg", "iframe"]):
        node.decompose()
    for node in soup.find_all("img"):
        src = node.get("src", "")
        if src.startswith("data:"):
            node.decompose()
    return re.sub(r"\s+", " ", str(soup))


@st.cache_resource(show_spinner="Loading ReaderLM-v2 (first run downloads the model)…")
def load_model(device_name: str):
    device, _ = get_device()
    # BF16 is used when supported; float32 is more broadly compatible with MPS/CPU.
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return tokenizer, model, device


def convert(html: str, max_new_tokens: int):
    device, _ = get_device()
    tokenizer, model, model_device = load_model(device.type)
    prompt = (
        "Extract the main content from the given HTML and convert it to Markdown format.\n"
        f"```html\n{html}\n```"
    )
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=32768)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


st.set_page_config(page_title="Webpage to Markdown · ReaderLM-v2", page_icon="📄", layout="wide")
st.title("Webpage to Markdown")
st.caption("Fetch a webpage and convert its HTML with the local Jina ReaderLM-v2 model.")

device, device_label = get_device()
st.info(f"Processing device: **{device_label}**. The model is downloaded from Hugging Face the first time you convert a page.")

with st.form("convert_form"):
    url = st.text_input("Website URL", value="https://www.elastic.co/", placeholder="https://www.elastic.co/")
    max_new_tokens = st.slider("Maximum Markdown output", min_value=512, max_value=8192, value=2048, step=256)
    submitted = st.form_submit_button("Convert to Markdown", type="primary")

if submitted:
    try:
        with st.spinner("Fetching webpage…"):
            raw_html = fetch_html(url.strip())
            html = clean_html(raw_html)
        if not html.strip():
            st.error("No HTML content was found at that URL.")
        else:
            st.caption(f"Prepared {len(html):,} characters of HTML for ReaderLM-v2.")
            with st.spinner("Converting with ReaderLM-v2…"):
                markdown = convert(html, max_new_tokens)
            if not markdown.strip():
                st.warning("The model returned empty output. Try a different page or a larger output limit.")
            else:
                st.subheader("Markdown")
                st.code(markdown, language="markdown", wrap_lines=True)
                st.download_button(
                    "Download Markdown",
                    data=markdown,
                    file_name="webpage.md",
                    mime="text/markdown",
                )
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not fetch the page: {exc}")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Conversion failed: {exc}")
        st.caption("Check the terminal for details. Model loading requires a compatible PyTorch/Transformers install and enough memory.")

with st.expander("About this app"):
    st.markdown(
        "The app downloads HTML directly from the URL you enter and sends it to ReaderLM-v2 running on this machine. "
        "HTML is capped at 300,000 characters before inference. The model is licensed under "
        "[CC BY-NC 4.0](https://huggingface.co/jinaai/ReaderLM-v2)."
    )
