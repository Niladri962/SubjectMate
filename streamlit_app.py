"""Local Streamlit interface (the Vercel deployment uses public/ + api/ instead).

    streamlit run streamlit_app.py
"""
import streamlit as st

from subjectmate import rag
from subjectmate.llm import PROVIDERS, LLMError, default_provider
from subjectmate.retriever import IndexNotBuiltError, load_meta

st.set_page_config(page_title="SubjectMate", page_icon="📘")
st.title("📘 SubjectMate")
st.caption("Answers come only from the course PDFs, with file names and page numbers.")

with st.sidebar:
    st.header("Model")
    ids = list(PROVIDERS)
    provider = st.selectbox("Provider", ids, index=ids.index(default_provider()),
                            format_func=lambda i: PROVIDERS[i].label)
    p = PROVIDERS[provider]
    model = st.text_input("Model", placeholder=p.default_model)
    api_key = st.text_input("API key", type="password",
                            placeholder="Using server key" if p.server_key() else "Required")
    k = st.slider("Passages to retrieve", 1, 10, 5)
    try:
        subjects = sorted({d["subject"] for d in load_meta()["documents"] if d.get("subject")})
    except IndexNotBuiltError:
        subjects = []
    subject = st.selectbox("Subject", [""] + subjects, format_func=lambda s: s.title() or "All subjects")
    st.header("Course material")
    try:
        for d in load_meta()["documents"]:
            st.write(f"- {d['name']}")
    except IndexNotBuiltError as exc:
        st.warning(str(exc))

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the course"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the course material…"):
                result = rag.answer(question, provider=provider, model=model, api_key=api_key, k=k,
                                    subject=subject or None)
            st.markdown(result["answer"])
            if result.get("note"):
                st.info(result["note"])
            if result["sources"]:
                with st.expander(f"Retrieved passages ({len(result['sources'])})"):
                    for s in result["sources"]:
                        st.markdown(f"**{s['source']}, {s['unit']} {s['page']}** · {s['subject']} · match {s['score']:.0%}")
                        st.caption(s["snippet"])
            st.session_state.history.append({"role": "assistant", "content": result["answer"]})
        except (IndexNotBuiltError, LLMError, RuntimeError) as exc:
            st.error(str(exc))
