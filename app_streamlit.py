import json
import os
import re

# Disables Streamlit's module file watcher that can trigger costly lazy imports
# inside transformers and slow down app reloads.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

from huggingface_hub import hf_hub_download
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

HF_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "julthia/lyrics-mood")
HF_REPO_TYPE = "model"
HF_MODEL_FILENAME = "best_model_bert_only.pt"
HF_TARGETS_FILENAME = "target_columns.json"

BERT_NAME = "bert-base-uncased"
BERT_MAX_LEN = 400
FAST_MAX_LEN = 192
PREDICTION_THRESHOLD = 0.49

EXAMPLE_LYRICS = {
    "Love Song": "Cause all of me\nLoves all of you\nLove your curves and all your edges\nAll your perfect imperfections",
    "Sad Mood": "And yet I fight, and yet I fight\nThis battle all alone\nNo one to cry to\nNo place to call home",
    "Motivational": "It's the eye of the tiger\nIT'S THE THRILL OF THE FIGHT\nRising up to the challenge of our rival",
}


def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"[a-z0-9]+/[a-z0-9]+", "", text)

    if text.startswith("lyrics "):
        text = text[len("lyrics ") :].strip()
    if text.startswith("written by "):
        text = text[len("written by ") :].strip()

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class BertMoodClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.pooler_output)
        return self.classifier(cls)


@st.cache_resource(show_spinner=False)
def load_tokenizer() -> BertTokenizer:
    return BertTokenizer.from_pretrained(BERT_NAME)


@st.cache_resource(show_spinner=False)
def load_target_columns() -> list[str]:
    targets_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_TARGETS_FILENAME,
        repo_type=HF_REPO_TYPE,
    )
    with open(targets_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model(num_labels: int) -> BertMoodClassifier:
    device = get_device()
    model = BertMoodClassifier(num_labels=num_labels)
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        repo_type=HF_REPO_TYPE,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_inputs(
    tokenizer: BertTokenizer,
    text: str,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def predict_labels(
    model: BertMoodClassifier,
    tokenizer: BertTokenizer,
    text: str,
    target_columns: list[str],
    max_len: int,
    device: torch.device,
) -> tuple[list[str], list[tuple[str, float]]]:
    cleaned = clean_lyrics(text)
    input_ids, attention_mask = prepare_inputs(tokenizer, cleaned, max_len=max_len)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        logits = model(input_ids, attention_mask)

        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted = [
        target_columns[i]
        for i, prob in enumerate(probs)
        if prob >= PREDICTION_THRESHOLD
    ]
    scored = sorted(
        [(target_columns[i], float(prob)) for i, prob in enumerate(probs)],
        key=lambda x: x[1],
        reverse=True,
    )
    return predicted, scored


def badge_style_for_label(label: str) -> tuple[str, str]:
    normalized = label.lower()
    if any(x in normalized for x in ["joy", "love", "party", "social"]):
        return "#7dd3fc", "😄"
    if any(x in normalized for x in ["sad", "fear"]):
        return "#fda4af", "😢"
    if any(x in normalized for x in ["exercise", "running"]):
        return "#86efac", "💪"
    if any(x in normalized for x in ["work", "study", "morning"]):
        return "#fcd34d", "📚"
    if "drive" in normalized:
        return "#c4b5fd", "🚗"
    return "#94a3b8", "🎧"


def render_result_badges(labels: list[str]) -> None:
    if not labels:
        return

    chips = []
    for label in labels:
        color, emoji = badge_style_for_label(label)
        chips.append(
            f"<span style='display:inline-block; margin:0.25rem 0.35rem 0.25rem 0; "
            f"padding:0.35rem 0.62rem; border-radius:999px; "
            f"background:rgba(15,23,42,0.9); border:1px solid {color}; color:{color}; "
            f"font-weight:600; font-size:0.9rem;'>{emoji} {label}</span>"
        )

    st.markdown("".join(chips), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Mood Prediction Demo", page_icon="🎵", layout="wide")

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1200px 700px at 10% -20%, rgba(61, 112, 173, 0.22), transparent 60%),
                radial-gradient(1000px 650px at 100% 0%, rgba(57, 180, 139, 0.18), transparent 62%),
                linear-gradient(180deg, #0c111b 0%, #101826 100%);
        }
        .hero {
            background: rgba(17, 24, 39, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.45);
        }
        .small-note {
            color: #c5d2e3;
            font-size: 0.95rem;
        }
        h1, h2, h3, h4, h5, h6, p, label, div, span {
            color: #e5edf8;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.82);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        [data-baseweb="textarea"] textarea {
            background-color: #0f172a;
            color: #e2e8f0;
            border: 1px solid #334155;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="slider"] {
            color: #e2e8f0;
        }
        [data-testid="stButton"] button {
            border-radius: 12px;
            border: 1px solid #3b475f;
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            color: #e5edf8;
        }
        [data-testid="stButton"] button:hover {
            border: 1px solid #6b7280;
            background: linear-gradient(135deg, #1e293b 0%, #273449 100%);
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0; color:#f5f7fb;">🎵 Mood Prediction for Song Lyrics</h2>
            <p style="margin:0.4rem 0 0 0;" class="small-note">
                Real-time demo using the BERT text model from Hugging Face Hub.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        target_columns = load_target_columns()
    except Exception as e:
        st.error(f"Could not download label metadata from HF repo '{HF_REPO_ID}': {e}")
        st.stop()

    tokenizer = load_tokenizer()
    device = get_device()

    if "lyrics_input" not in st.session_state:
        st.session_state.lyrics_input = ""

    with st.sidebar:
        st.header("⚙️ Prediction Settings")
        max_len = FAST_MAX_LEN
        st.caption(f"Max sequence length: {max_len} tokens (fixed)")
        st.caption("Model: Text-only BERT")
        st.caption(f"Source: HF Hub ({HF_REPO_ID})")

        st.divider()
        st.subheader("📝 Quick Examples")
        st.caption("Click any button to auto-fill realistic sample lyrics!")
        for idx, (label, text) in enumerate(EXAMPLE_LYRICS.items(), start=1):
            if st.button(label, key=f"sample_{idx}", use_container_width=True):
                st.session_state.lyrics_input = text

    st.caption("✨ Text-only mode is enabled. Paste lyrics and run prediction.")

    lyrics = st.text_area(
        "Paste song lyrics",
        height=220,
        placeholder="Type or paste song lyrics here...",
        key="lyrics_input",
    )

    if st.button("🔮 Run Prediction", type="primary", use_container_width=True):
        if not lyrics.strip():
            st.warning("Please enter song lyrics.")
            st.stop()

        with st.status("Running prediction...", expanded=False) as status:
            status.write("Loading text model")
            try:
                model = load_model(num_labels=len(target_columns))
            except Exception as e:
                st.error(f"Could not download/load model from HF repo '{HF_REPO_ID}': {e}")
                st.stop()
            status.write("Tokenizing lyrics and running inference")

            predicted, scored = predict_labels(
                model=model,
                tokenizer=tokenizer,
                text=lyrics,
                target_columns=target_columns,
                max_len=max_len,
                device=device,
            )
            status.update(label="Prediction complete", state="complete")

        st.subheader("🎯 Result")
        if predicted:
            st.success("Top moods found:")
            render_result_badges(predicted)
        else:
            st.info("No labels passed the fixed decision threshold.")

        st.subheader("📊 Top 8 labels")
        top_k = scored[:8]
        st.table({"label": [x[0] for x in top_k], "probability": [round(x[1], 4) for x in top_k]})


if __name__ == "__main__":
    main()
