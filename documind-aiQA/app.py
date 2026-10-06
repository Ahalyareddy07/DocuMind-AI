"""DocuMind AI - Intelligent Document Question Answering System (Streamlit UI)."""
import html
import logging
import re
from pathlib import Path

import streamlit as st

from modules import chat_manager as cm
from modules.document_processor import DocumentError, chunk_pages, compute_stats, extract_pages
from modules.embeddings import EmbeddingError, get_embedder
from modules.quiz_generator import TYPES, generate_quiz
from modules.rag_pipeline import LLMUnavailable, answer_question, list_models
from modules.summarizer import FORMATS, summarize
from modules.vector_store import VectorStore
from utils import file_utils as fu
from utils.helpers import ROOT, human_size, load_settings, new_id, now, save_settings

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")
S = st.session_state
PAGES = ["🏠 Dashboard", "📚 My Documents", "💬 AI Chat", "📝 Summarize", "🎯 Quiz Generator",
         "🔍 Search", "📜 Chat History", "⚙️ Settings"]
DASH, DOCS, CHAT, SUM, QUIZ, SEARCH, HIST, SET = PAGES
EXAMPLES = ["What is this document about?", "Give me the key points.",
            "Explain the main concept in simple words.", "What is the difference between 3NF and BCNF?"]


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_data(ttl=10)
def installed_models(url):
    return list_models(url)


store, cfg = get_store(), load_settings()


# ------------------------------------------------------------------ callbacks
def go(p): S.page = p
def new_chat(): S.cur_chat, S.chat_ver = None, S.get("chat_ver", 0) + 1
def ask(q): S.pending_q = q
def toggle_open(d): S.open_doc = None if S.get("open_doc") == d else d


def open_chat(cid):
    S.cur_chat, S.chat_ver, S.page = cid, S.get("chat_ver", 0) + 1, CHAT


def act(page, doc_id):
    """Jump from the document library to a feature pre-selecting that document."""
    S.page = page
    if page == CHAT:
        S.chat_scope, S.chat_sel = "Selected Documents", [doc_id]
    elif page == QUIZ:
        S.quiz_scope, S.quiz_sel = "Selected Documents", [doc_id]
    elif page == SUM:
        S.sum_doc = doc_id


def remove_doc(doc_id):
    store.remove_document(doc_id)
    fu.delete_doc(doc_id)


# ------------------------------------------------------------------ theme
def inject_css(theme):
    d = theme == "dark"
    bg, card, fg, mut, bd, side = (("#0f1220", "#181c2f", "#e8eaf6", "#9aa3c7", "#2a3050", "#12162a") if d
                                   else ("#f5f7fb", "#ffffff", "#1b2040", "#6b7494", "#e3e7f2", "#ffffff"))
    st.markdown(f"""<style>
    .stApp{{background:{bg};transition:background .3s,color .3s}}
    .stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp p,.stApp label,.stApp li,.stApp td,.stApp th,
    .stApp [data-testid="stMetricValue"],.stApp [data-testid="stCaptionContainer"]{{color:{fg}}}
    section[data-testid="stSidebar"]{{background:{side};border-right:1px solid {bd}}}
    header[data-testid="stHeader"]{{background:transparent}}
    div[data-testid="stBottom"]>div{{background:{bg}}}
    .stTextInput input,.stTextArea textarea,.stNumberInput input,[data-baseweb="select"]>div,
    [data-testid="stChatInput"] textarea{{background:{card}!important;color:{fg}!important;border-radius:10px}}
    div[data-testid="stExpander"],div[data-testid="stChatMessage"],div[data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"]{{background:{card};border-color:{bd};border-radius:14px}}
    .stButton>button,.stDownloadButton>button{{border-radius:10px;border:1px solid {bd};background:{card};
      color:{fg};transition:all .15s}}
    .stButton>button:hover{{transform:translateY(-1px);border-color:#6c5ce7;color:#6c5ce7;
      box-shadow:0 4px 14px rgba(108,92,231,.25)}}
    .stButton>button[kind="primary"]{{background:linear-gradient(135deg,#6c5ce7,#a259ff);color:#fff;border:0}}
    .stButton>button[kind="primary"] p{{color:#fff}}
    .card{{background:{card};border:1px solid {bd};border-radius:16px;padding:18px 20px;
      box-shadow:0 4px 18px rgba(0,0,0,.12);transition:transform .15s}}
    .card:hover{{transform:translateY(-2px)}}
    .card .v{{font-size:1.9rem;font-weight:700;color:{fg}}} .card .l{{color:{mut};font-size:.85rem}}
    .empty{{text-align:center;padding:40px;font-size:1.1rem;color:{fg}}}
    .hero{{background:linear-gradient(135deg,#6c5ce7,#a259ff 60%,#00b8d9);border-radius:18px;
      padding:26px 30px;color:#fff;margin-bottom:18px}} .hero h1,.hero p{{color:#fff!important;margin:0}}
    .topbar{{display:flex;justify-content:space-between;align-items:center;padding:6px 0 14px;color:{fg}}}
    .chip{{background:{card};border:1px solid {bd};border-radius:99px;padding:4px 14px;font-size:.85rem;color:{fg}}}
    .badge{{border-radius:99px;padding:2px 10px;font-size:.78rem;font-weight:600;color:#fff}}
    mark{{background:#ffd54f;color:#000;border-radius:3px;padding:0 2px}}
    @media(max-width:700px){{.card .v{{font-size:1.4rem}}.hero{{padding:18px}}}}
    </style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ small UI helpers
def stat_card(icon, label, value):
    st.markdown(f'<div class="card"><div class="l">{icon} {label}</div><div class="v">{value}</div></div>',
                unsafe_allow_html=True)


def badge(status):
    colour = {"Ready": "#00b894", "Processing": "#fdcb6e", "Uploading": "#0984e3", "Error": "#d63031"}.get(status, "#636e72")
    return f'<span class="badge" style="background:{colour}">{status}</span>'


def empty(msg, target=DOCS, label="Go to My Documents"):
    st.markdown(f'<div class="card empty">📭<br><b>{msg}</b></div>', unsafe_allow_html=True)
    st.button(label, on_click=go, args=(target,))


def ready_docs():
    return [d for d in fu.load_docs().values() if d.get("status") == "Ready"]


def doc_selector(prefix, ready, label="Ask about:"):
    """All documents / one / several. Returns the selected document ids."""
    names = {d["id"]: d["name"] for d in ready}
    scope = st.radio(label, ["All Documents", "Selected Documents"], horizontal=True, key=f"{prefix}_scope")
    if scope == "All Documents":
        return list(names)
    S[f"{prefix}_sel"] = [x for x in S.get(f"{prefix}_sel", []) if x in names]
    return st.multiselect("Choose one or more documents", list(names), format_func=names.get, key=f"{prefix}_sel")


def show_sources(sources):
    if sources:
        st.markdown("**Sources**")
    for s in sources:
        with st.expander(f"📄 {s['doc_name']} — Page {s['page']}"):
            st.write(s["text"])


def run_ai(spinner, fn, *a, **k):
    """Run an AI task and turn every failure into a friendly message (no stack traces)."""
    try:
        with st.spinner(spinner):
            return fn(*a, **k)
    except (ValueError, LLMUnavailable, EmbeddingError) as e:
        st.error(f"⚠️ {e}")
    except Exception:
        logging.exception("AI task failed")
        st.error("⚠️ Something went wrong. Please try again.")
    return None


# ------------------------------------------------------------------ document pipeline
def process_upload(name, data, bar=None):
    """Validate -> extract -> clean -> chunk -> embed -> store. Returns (ok, message)."""
    try:
        fu.validate_upload(name, data, cfg["max_upload_mb"])
    except ValueError as e:
        return False, f"❌ {name}: {e}"
    doc_id, safe = new_id(), fu.sanitize_filename(name)
    path = fu.save_upload(doc_id, safe, data)
    fu.add_doc({"id": doc_id, "name": safe, "type": Path(safe).suffix[1:].upper(), "size": len(data),
                "uploaded": now(), "status": "Processing", "path": path.name})

    def step(p, t):
        if bar:
            bar.progress(p, text=f"{safe}: {t}")
    try:
        step(.15, "extracting text…")
        pages = extract_pages(path)
        step(.35, "chunking…")
        chunks = chunk_pages(doc_id, safe, pages, cfg["chunk_size"], cfg["chunk_overlap"])
        step(.55, "creating embeddings…")
        emb = get_embedder(cfg["embedding_model"])
        vecs = emb.encode([c["text"] for c in chunks])
        step(.85, "saving to vector database…")
        store.add(vecs, chunks, emb.name)
        stats = compute_stats(pages)
        fu.update_doc(doc_id, status="Ready", chunks=len(chunks), preview=pages[0][1][:1200], **stats)
        step(1.0, "ready ✓")
        return True, f"✅ {safe} is ready ({stats['pages']} pages, {len(chunks)} chunks)."
    except (DocumentError, EmbeddingError, ValueError) as e:
        msg = str(e)
    except Exception:
        logging.exception("Processing failed")
        msg = "Unexpected error while processing this document."
    fu.update_doc(doc_id, status="Error", error=msg)
    return False, f"❌ {safe}: {msg}"


def load_sample():
    return process_upload("DBMS_Normalization_Sample.txt", (ROOT / "data" / "sample_document.txt").read_bytes())


# ------------------------------------------------------------------ pages
def page_dashboard():
    docs = list(fu.load_docs().values())
    st.markdown('<div class="hero"><h1>👋 Welcome to DocuMind AI</h1>'
                '<p>Chat with your documents. Understand information instantly.</p></div>', unsafe_allow_html=True)
    c = st.columns(4)
    with c[0]: stat_card("📄", "Documents", len(docs))
    with c[1]: stat_card("📑", "Total pages", sum(d.get("pages", 0) for d in docs))
    with c[2]: stat_card("💾", "Total size", human_size(sum(d["size"] for d in docs)))
    with c[3]: stat_card("❓", "Questions asked", cm.count_questions())
    st.subheader("⚡ Quick actions")
    q = st.columns(4)
    q[0].button("⬆️ Upload Document", on_click=go, args=(DOCS,), type="primary")
    q[1].button("💬 Start Chat", on_click=go, args=(CHAT,))
    q[2].button("📝 Summarize Document", on_click=go, args=(SUM,))
    q[3].button("🎯 Generate Quiz", on_click=go, args=(QUIZ,))
    st.subheader("🕒 Recent documents")
    if not docs:
        st.markdown('<div class="card empty">📭<br><b>No documents yet.</b> Upload one, or try the sample.</div>',
                    unsafe_allow_html=True)
        if st.button("📥 Load sample document"):
            with st.spinner("Processing sample…"):
                ok, msg = load_sample()
            (st.success if ok else st.error)(msg)
            if ok: st.rerun()
        return
    st.dataframe([{"Name": d["name"], "Type": d["type"], "Size": human_size(d["size"]),
                   "Pages": d.get("pages", "-"), "Uploaded": d["uploaded"], "Status": d["status"]}
                  for d in reversed(docs)][:5], hide_index=True)


def page_documents():
    st.header("📚 My Documents")
    tab_up, tab_lib = st.tabs(["⬆️ Upload", "📄 Library"])
    with tab_up:
        files = st.file_uploader("Drag & drop PDF, DOCX or TXT files here, or browse",
                                 type=["pdf", "docx", "txt"], accept_multiple_files=True,
                                 key=f"up{S.get('up_ver', 0)}", help=f"Max {cfg['max_upload_mb']} MB per file")
        c1, c2 = st.columns([1, 1])
        if c1.button("🚀 Process documents", type="primary", disabled=not files):
            for f in files:
                bar = st.progress(0, text=f"{f.name}: uploading…")
                ok, msg = process_upload(f.name, f.getvalue(), bar)
                (st.success if ok else st.error)(msg)
            S.up_ver = S.get("up_ver", 0) + 1  # reset the uploader on next run
        if c2.button("📥 Load sample document"):
            with st.spinner("Processing sample…"):
                ok, msg = load_sample()
            (st.success if ok else st.error)(msg)
    with tab_lib:
        docs = list(reversed(list(fu.load_docs().values())))
        term = st.text_input("🔎 Search documents", placeholder="Filter by file name…").lower()
        docs = [d for d in docs if term in d["name"].lower()]
        if not docs:
            st.markdown('<div class="card empty">📭<br><b>No documents found.</b></div>', unsafe_allow_html=True)
        for d in docs:
            with st.container(border=True):
                st.markdown(f"**📄 {html.escape(d['name'])}** &nbsp; {badge(d['status'])}<br>"
                            f"<span style='opacity:.7;font-size:.85rem'>{d['type']} · {human_size(d['size'])} · "
                            f"{d.get('pages', '-')} pages · uploaded {d['uploaded']}</span>", unsafe_allow_html=True)
                if d["status"] == "Error":
                    st.error(d.get("error", "Processing failed."))
                b = st.columns(5)
                b[0].button("📂 Open", key=f"o{d['id']}", on_click=toggle_open, args=(d["id"],))
                ok = d["status"] == "Ready"
                b[1].button("💬 Chat", key=f"c{d['id']}", on_click=act, args=(CHAT, d["id"]), disabled=not ok)
                b[2].button("📝 Summarize", key=f"s{d['id']}", on_click=act, args=(SUM, d["id"]), disabled=not ok)
                b[3].button("🎯 Quiz", key=f"q{d['id']}", on_click=act, args=(QUIZ, d["id"]), disabled=not ok)
                b[4].button("🗑️ Delete", key=f"d{d['id']}", on_click=remove_doc, args=(d["id"],))
                if S.get("open_doc") == d["id"] and ok:
                    m = st.columns(6)
                    for col, (lab, val) in zip(m, [("Pages", d["pages"]), ("Words", f"{d['words']:,}"),
                                                   ("Characters", f"{d['characters']:,}"),
                                                   ("Paragraphs", d["paragraphs"]), ("File size", human_size(d["size"])),
                                                   ("Reading time", f"{d['reading_minutes']} min")]):
                        col.metric(lab, val)
                    st.markdown("**Preview**")
                    st.text(d.get("preview", "")[:1000])


def render_msg(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources", []))


def page_chat():
    st.header("💬 AI Chat")
    ready = ready_docs()
    if not ready:
        return empty("Upload a document first to start chatting.")
    ids = doc_selector("chat", ready)
    chats = cm.list_chats()
    opts = [None] + [c["id"] for c in chats]
    titles = {c["id"]: f"📄 {c['title']} · {c['updated']}" for c in chats}
    cur = S.get("cur_chat")
    c1, c2 = st.columns([4, 1])
    sel = c1.selectbox("Conversation", opts, index=opts.index(cur) if cur in opts else 0,
                       format_func=lambda x: "➕ New chat" if x is None else titles[x],
                       key=f"chatsel_{S.get('chat_ver', 0)}", label_visibility="collapsed")
    S.cur_chat = sel
    c2.button("➕ New chat", on_click=new_chat)
    msgs = cm.get_chat(sel)["messages"] if sel else []
    for m in msgs:
        render_msg(m)
    if not msgs:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(2)
        for i, e in enumerate(EXAMPLES):
            cols[i % 2].button(f"💡 {e}", key=f"ex{i}", on_click=ask, args=(e,))
    q = st.chat_input("Ask a question about your documents…") or S.pop("pending_q", None)
    if not q:
        return
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        res = run_ai("Searching your documents and thinking…", answer_question, q, store,
                     get_embedder(cfg["embedding_model"]), cfg, ids, msgs)
        if not res:
            return
        st.markdown(res["answer"])
        show_sources(res["sources"])
    if not sel:
        sel = cm.create_chat(q[:45])
        S.cur_chat, S.chat_ver = sel, S.get("chat_ver", 0) + 1
    srcs = [{k: s[k] for k in ("doc_name", "page", "text", "score")} for s in res["sources"]]
    cm.add_messages(sel, [{"role": "user", "content": q},
                          {"role": "assistant", "content": res["answer"], "sources": srcs}])


def page_summarize():
    st.header("📝 Summarize")
    ready = ready_docs()
    if not ready:
        return empty("Upload a document to summarize it.")
    names = {d["id"]: d["name"] for d in ready}
    if S.get("sum_doc") not in names:
        S.sum_doc = list(names)[0]
    doc = st.selectbox("Document", list(names), format_func=names.get, key="sum_doc")
    c1, c2 = st.columns(2)
    kind = c1.selectbox("Summary type", list(FORMATS))
    length = c2.select_slider("Length", ["Short", "Medium", "Detailed"], value="Medium")
    if st.button("✨ Generate summary", type="primary"):
        out = run_ai("Reading the document and writing the summary…", summarize, doc, store, cfg, kind, length)
        if out:
            S.summary = {"name": names[doc], "kind": kind, "text": out}
    s = S.get("summary")
    if s:
        st.markdown(f"### {s['kind']} — {s['name']}")
        st.markdown(s["text"])
        st.download_button("⬇️ Download as .md", s["text"], file_name=f"summary_{s['name']}.md")


def page_quiz():
    st.header("🎯 Quiz Generator")
    ready = ready_docs()
    if not ready:
        return empty("Upload a document to generate a quiz.")
    ids = doc_selector("quiz", ready, "Create quiz from:")
    c1, c2 = st.columns(2)
    n = c1.selectbox("Number of questions", [5, 10, 15, 20])
    qtype = c2.selectbox("Question type", list(TYPES))
    if st.button("🎯 Generate quiz", type="primary"):
        qs = run_ai("Writing your quiz (this can take a minute)…", generate_quiz, ids, store, cfg, n, qtype)
        if qs:
            S.quiz = {"qs": qs, "id": new_id(), "graded": None}
    quiz = S.get("quiz")
    if not quiz:
        return
    with st.form("quiz_form"):
        for i, q in enumerate(quiz["qs"]):
            key, label = f"qz{quiz['id']}_{i}", f"**Q{i + 1}. {q['question']}**"
            st.markdown(label)
            if q["type"] == "mcq":
                st.radio("Choose", [f"{l}. {o}" for l, o in zip("ABCD", q["options"])], index=None,
                         key=key, label_visibility="collapsed")
            elif q["type"] == "tf":
                st.radio("Choose", ["True", "False"], index=None, key=key, label_visibility="collapsed")
            else:
                st.text_input("Your answer", key=key, label_visibility="collapsed")
        submitted = st.form_submit_button("✅ Submit answers", type="primary")
    if submitted:
        graded = []
        for i, q in enumerate(quiz["qs"]):
            user = S.get(f"qz{quiz['id']}_{i}") or ""
            if q["type"] == "mcq":
                ok = user[:1] == q["answer"]
            elif q["type"] == "tf":
                ok = user == q["answer"]
            else:  # short answer: semantic similarity between the user's and the model answer
                try:
                    v = get_embedder(cfg["embedding_model"]).encode([user, q["answer"]]) if user.strip() else None
                    ok = bool(v is not None and float(v[0] @ v[1]) >= 0.55)
                except EmbeddingError:
                    ok = False
            graded.append((ok, user))
        quiz["graded"] = graded
    if quiz["graded"]:
        score = sum(g[0] for g in quiz["graded"])
        st.success(f"## Score: {score}/{len(quiz['qs'])}")
        for i, (q, (ok, user)) in enumerate(zip(quiz["qs"], quiz["graded"]), 1):
            correct = q["answer"] if q["type"] != "mcq" else f"{q['answer']}. {q['options']['ABCD'.index(q['answer'])]}"
            with st.expander(f"{'✅' if ok else '❌'} Q{i}. {q['question']}"):
                st.write(f"**Your answer:** {user or '—'}")
                st.write(f"**Correct answer:** {correct}")
                if q["explanation"]:
                    st.write(f"**Explanation:** {q['explanation']}")


def page_search():
    st.header("🔍 Search Inside Documents")
    ready = ready_docs()
    if not ready:
        return empty("Upload a document to search it.")
    ids = set(doc_selector("search", ready, "Search in:"))
    term = st.text_input("Keyword or phrase", placeholder="e.g. Normalization").strip()
    if not term:
        return st.info("Type a keyword to find matching sections.")
    hits = [c for c in store.chunks if c["doc_id"] in ids and term.lower() in c["text"].lower()]
    st.caption(f"{len(hits)} matching section(s)")
    for c in hits[:50]:
        i = c["text"].lower().find(term.lower())
        snip = c["text"][max(0, i - 150): i + len(term) + 250]
        snip = re.sub(re.escape(html.escape(term)), lambda m: f"<mark>{m.group(0)}</mark>",
                      html.escape(snip), flags=re.I)
        st.markdown(f'<div class="card" style="margin-bottom:10px"><b>📄 {html.escape(c["doc_name"])} — '
                    f'Page {c["page"]}</b><br>…{snip}…</div>', unsafe_allow_html=True)


def page_history():
    st.header("📜 Chat History")
    chats = cm.list_chats()
    if not chats:
        return empty("No conversations yet.", CHAT, "Start a chat")
    for c in chats:
        with st.container(border=True):
            st.markdown(f"**📄 {html.escape(c['title'])}**  \n{c['updated']} · {len(c['messages'])} messages")
            b = st.columns([1, 1, 3])
            b[0].button("▶️ Continue", key=f"oc{c['id']}", on_click=open_chat, args=(c["id"],))
            b[1].button("🗑️ Delete", key=f"dc{c['id']}", on_click=cm.delete_chat, args=(c["id"],))
            with b[2].popover("✏️ Rename"):
                t = st.text_input("New title", value=c["title"], key=f"rt{c['id']}")
                if st.button("Save", key=f"rs{c['id']}"):
                    cm.rename_chat(c["id"], t)
                    st.rerun()


def page_settings():
    st.header("⚙️ Settings")
    ms = installed_models(cfg["ollama_base_url"])
    st.caption("Installed Ollama models: " + (", ".join(ms) if ms else "Ollama not reachable"))
    with st.form("settings"):
        st.subheader("AI settings")
        a, b = st.columns(2)
        url = a.text_input("Ollama URL", cfg["ollama_base_url"])
        model = b.text_input("AI model", cfg["model"])
        temp = a.slider("Temperature", 0.0, 1.0, float(cfg["temperature"]), 0.05)
        topk = b.slider("Retrieved chunks (top-k)", 1, 15, int(cfg["top_k"]))
        emb = a.text_input("Embedding model", cfg["embedding_model"])
        st.subheader("Document settings")
        c1, c2, c3 = st.columns(3)
        size = c1.number_input("Chunk size (chars)", 200, 4000, int(cfg["chunk_size"]), 50)
        ovl = c2.number_input("Chunk overlap", 0, 1000, int(cfg["chunk_overlap"]), 25)
        mx = c3.number_input("Max upload size (MB)", 1, 200, int(cfg["max_upload_mb"]))
        if st.form_submit_button("💾 Save settings", type="primary"):
            if ovl >= size:
                st.error("Chunk overlap must be smaller than chunk size.")
            else:
                cfg.update(ollama_base_url=url.rstrip("/"), model=model.strip(), temperature=temp, top_k=topk,
                           embedding_model=emb.strip(), chunk_size=int(size), chunk_overlap=int(ovl),
                           max_upload_mb=int(mx))
                save_settings(cfg)
                st.success("Settings saved. Chunk settings apply to newly uploaded documents.")
    if store.model and store.model != cfg["embedding_model"]:
        st.warning("The embedding model differs from the one used to build the index.")
        if st.button("🔄 Rebuild index with the new embedding model"):
            if run_ai("Re-embedding all chunks…", lambda: store.reindex(get_embedder(cfg["embedding_model"])) or True):
                st.success("Index rebuilt.")
    st.subheader("Interface & data")
    st.caption("Dark/Light mode: use the toggle in the sidebar.")
    d1, d2 = st.columns(2)
    if d1.button("🗑️ Clear chat history"):
        cm.clear_chats()
        S.cur_chat = None
        st.success("Chat history cleared.")
    sure = d2.checkbox("I understand this deletes every document")
    if d2.button("🗑️ Clear all documents", disabled=not sure):
        store.clear()
        fu.clear_docs()
        st.success("All documents removed.")


# ------------------------------------------------------------------ layout
S.setdefault("page", PAGES[0])
with st.sidebar:
    st.markdown("## 🧠 DocuMind AI")
    st.caption("Chat with your documents.")
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    dark = st.toggle("🌙 Dark / ☀️ Light mode", value=cfg["theme"] == "dark")
    if dark != (cfg["theme"] == "dark"):
        cfg["theme"] = "dark" if dark else "light"
        save_settings(cfg)
inject_css(cfg["theme"])

ms = installed_models(cfg["ollama_base_url"])
if ms is None:
    status = "🔴 Ollama offline"
elif cfg["model"] in ms or any(m.split(":")[0] == cfg["model"] for m in ms):
    status = f"🟢 {cfg['model']}"
else:
    status = f"🟡 {cfg['model']} not installed"
st.markdown(f'<div class="topbar"><div><b>DocuMind AI</b> · <span style="opacity:.7">Chat with your documents. '
            f'Understand information instantly.</span></div><div class="chip">{status}</div></div>',
            unsafe_allow_html=True)
if ms is None:
    st.warning(f"Local AI model not reachable. Install Ollama (https://ollama.com), then run "
               f"`ollama pull {cfg['model']}`. Uploading and searching still work without it.")

{DASH: page_dashboard, DOCS: page_documents, CHAT: page_chat, SUM: page_summarize, QUIZ: page_quiz,
 SEARCH: page_search, HIST: page_history, SET: page_settings}[S.page]()
