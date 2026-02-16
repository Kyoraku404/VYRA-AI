import re
import shutil
from pathlib import Path

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import TokenTextSplitter


# =========================
# CONFIG
# =========================
LLM_MODEL = "granite4:3b"        # chat model
EMBED_MODEL = "nomic-embed-text"    # embeddings model

TEMPERATURE = 0.35
TOP_K = 3

# Safer chunking for embeddings (prevents 400 context-length)
CHUNK_TOKENS = 180
OVERLAP_TOKENS = 35

# Absolute safety clamp (chars) before embedding + before sending context to LLM
MAX_NODE_CHARS = 1200
MAX_CTX_CHARS = 5000


# =========================
# PATHS (root/Data + root/Storage)
# =========================
def get_dirs():
    app_dir = Path(__file__).resolve().parent      # .../valorant-ai/app
    root_dir = app_dir.parent                      # .../valorant-ai

    data_dir = root_dir / "Data"
    storage_dir = root_dir / "Storage"

    data_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    return root_dir, data_dir, storage_dir


ROOT_DIR, DATA_DIR, STORAGE_DIR = get_dirs()


# =========================
# OLLAMA MODELS
# =========================
def configure_models():
    # LLM
    Settings.llm = Ollama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        request_timeout=120,
        keep_alive="10m",
        additional_kwargs={
            "num_ctx": 2048,
            "num_predict": 320,
        },
    )

    # Embeddings
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        keep_alive="10m",
    )


# =========================
# HELPERS
# =========================
def clamp_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s[:max_chars]


def detect_lang(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


def system_style(lang: str) -> str:
    if lang == "ar":
        return (
            "أنت مساعد ذكي.\n"
            "إذا كان سؤال المستخدم بالعربية فأجب بالعربية الفصحى، وإذا كان بالإنجليزية فأجب بالإنجليزية.\n"
            "أجب بشكل طبيعي بدون عناوين ثابتة أو مقدمات طويلة.\n"
            "اجعل الجواب عملياً ومختصراً.\n"
        )
    return (
        "You are a smart assistant.\n"
        "If the user writes Arabic, answer in Modern Standard Arabic. If English, answer in English.\n"
        "Be natural (no scripted headings). Keep it practical and concise.\n"
    )


def is_valorant_related(q: str) -> bool:
    k = q.lower()
    keys = [
        "valorant", "vct", "rank", "rr", "elo",
        "aim", "crosshair", "peek", "spray", "burst", "tracking", "flick",
        "agent", "duelist", "initiator", "controller", "sentinel",
        "map", "haven", "bind", "ascent", "split", "lotus", "breeze", "sunset", "icebox",
        "plant", "defuse", "eco", "buy", "save", "force"
    ]
    return any(x in k for x in keys)


def storage_has_index(storage_dir: Path) -> bool:
    return (storage_dir / "docstore.json").exists() or (storage_dir / "index_store.json").exists()


# =========================
# INDEX (build once + persist)
# =========================
def build_or_load_index():
    # If Storage already has index -> load (fast)
    if storage_has_index(STORAGE_DIR):
        print("📦 Loading existing index from Storage...")
        sc = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
        return load_index_from_storage(sc)

    # First time: build
    print("🧱 Building index (first time only)...")

    if not any(DATA_DIR.glob("*")):
        print(f"⚠️ Data folder empty: {DATA_DIR}")
        print("➡️ Put your PDF/TXT files inside Data/ then run again.")
        return None

    docs = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    print(f"✅ Loaded docs: {len(docs)}")

    splitter = TokenTextSplitter(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
    )

    nodes = splitter.get_nodes_from_documents(docs)

    # hard clamp each node to avoid ollama embed 400
    for n in nodes:
        n.set_content(clamp_text(n.get_content(), MAX_NODE_CHARS))

    print(f"✅ Built nodes: {len(nodes)}")

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    print("✅ Index saved to Storage.")
    return index


def retrieve_context(index, q: str) -> str:
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    res = retriever.retrieve(q)
    ctx = "\n\n".join([r.node.get_content() for r in res])
    return clamp_text(ctx, MAX_CTX_CHARS)


# =========================
# ANSWER (unified: Valorant uses Data, others general)
# =========================
def answer(index, q: str) -> str:
    lang = detect_lang(q)
    s = system_style(lang)

    # quick greeting
    if q.strip().lower() in ("hi", "hello", "hey", "salam", "سلام"):
        return "مرحبًا 👋 كيف أساعدك؟" if lang == "ar" else "Yo 👋 How can I help?"

    if index is not None and is_valorant_related(q):
        ctx = retrieve_context(index, q)
        prompt = f"""{s}

Use the context ONLY if it helps. If not needed, ignore it.

Context:
{ctx}

User question:
{q}

Answer now (natural + practical).
"""
    else:
        prompt = f"""{s}

User question:
{q}

Answer now (natural + practical).
"""

    resp = Settings.llm.complete(prompt)
    return str(resp).strip()


# =========================
# MAIN LOOP
# =========================
def main():
    configure_models()

    index = build_or_load_index()  # may be None if no Data files

    while True:
        try:
            q = input("\n🎮 Ask (or 'exit' / 'rebuild'): ").strip()
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break

        if not q:
            continue

        if q.lower() == "exit":
            print("👋 Bye!")
            break

        if q.lower() == "rebuild":
            # delete Storage and rebuild index
            if STORAGE_DIR.exists():
                shutil.rmtree(STORAGE_DIR)
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            index = build_or_load_index()
            continue

        try:
            print("\n🧠 Answer:\n", answer(index, q))
        except Exception as e:
            msg = str(e)
            if "Failed to connect to Ollama" in msg:
                print("⚠️ شغّل Ollama أولاً:  ollama serve")
            elif "context length" in msg or "exceeds the context length" in msg:
                print("⚠️ كاين نص طويل بزاف. جرّب rebuild وخلّي PDF يكون نص عادي (ماشي جدول كبير).")
            else:
                print("⚠️ Error:", e)


if __name__ == "__main__":
    main()
