"""Section F — Streamlit demo UI.

Chat-style Q&A over the indexed SEC filings with:
  - Sidebar metadata filters (company, year, filing type)
  - Toggles for query rewriter, reranker, debug view
  - Inline citations with expandable source cards
  - Status indicators per pipeline stage
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.indexing import collection_stats, get_collection
from src.rag_chain import answer


st.set_page_config(page_title="Financial Filings Q&A (RAG)", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner=False)
def warmup():
    """Warm up the embedder and Chroma client once per session."""
    return collection_stats()


@st.cache_data(show_spinner=False, ttl=300)
def get_metadata_options() -> dict:
    """Pull distinct values for company / year / filing_type from the collection."""
    coll = get_collection()
    if coll.count() == 0:
        return {"companies": [], "years": [], "filing_types": []}
    sample = coll.get(include=["metadatas"], limit=min(coll.count(), 5000))
    metas = sample.get("metadatas") or []
    companies, years, filings = set(), set(), set()
    for m in metas:
        if m.get("company"):
            companies.add(str(m["company"]))
        if m.get("year") not in (None, ""):
            try:
                years.add(int(m["year"]))
            except Exception:
                pass
        if m.get("filing_type"):
            filings.add(str(m["filing_type"]))
    return {
        "companies": sorted(companies),
        "years": sorted(years, reverse=True),
        "filing_types": sorted(filings),
    }


def render_citation(c: dict, used_in_answer: bool) -> None:
    badge = "✅ cited" if used_in_answer else "•"
    header_bits = [c.get("company") or c.get("source") or "filing"]
    if c.get("year"):
        header_bits.append(str(c["year"]))
    if c.get("section"):
        header_bits.append(c["section"])
    if c.get("page"):
        header_bits.append(f"p.{c['page']}")
    header = " · ".join(str(b) for b in header_bits if b)

    with st.expander(f"[chunk_{c['index']}]  {header}  ·  score {c['score']:.3f}  {badge}"):
        st.caption(f"`{c.get('source', '')}` · chunk_id `{c.get('chunk_id', '')}`")
        st.write(c["text"])


# ---------- sidebar ----------

with st.sidebar:
    st.title("📊 Financial RAG")
    stats = warmup()
    st.metric("Indexed chunks", stats["count"])
    st.caption(f"`{settings.COLLECTION_NAME}` · `{settings.EMBED_MODEL}`")
    st.divider()

    st.subheader("Filters")
    options = get_metadata_options()
    selected_company = st.selectbox("Company", ["(any)"] + options["companies"])
    selected_year = st.selectbox("Year", ["(any)"] + [str(y) for y in options["years"]])
    selected_filing = st.selectbox("Filing type", ["(any)"] + options["filing_types"])

    st.divider()
    st.subheader("Pipeline")
    use_rewriter = st.toggle("Query rewriter (HyDE)", value=settings.USE_QUERY_REWRITER)
    use_reranker = st.toggle("Cross-encoder reranker", value=settings.USE_RERANKER)
    show_debug = st.toggle("Show retrieval debug", value=False)

    st.divider()
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.rerun()


filters = {}
if selected_company != "(any)":
    filters["company"] = selected_company
if selected_year != "(any)":
    filters["year"] = int(selected_year)
if selected_filing != "(any)":
    filters["filing_type"] = selected_filing

# ---------- main ----------

st.title("Financial Filings Q&A")
st.caption(
    "Grounded answers from SEC 10-K / 10-Q filings. "
    "Every claim is cited to a retrieved chunk. Out-of-scope and insufficient-context queries are refused."
)

if stats["count"] == 0:
    st.warning(
        "No chunks are indexed yet. Drop SEC filings into `data/raw/` and run "
        "`python scripts/build_index.py --rebuild` from the project root."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("response"):
            r = msg["response"]
            cited = set(r["citation_check"]["cited_indices"])
            if r["citations"]:
                st.caption("**Sources**")
                for c in r["citations"]:
                    render_citation(c, used_in_answer=c["index"] in cited)
            if show_debug:
                with st.expander("🔍 Debug"):
                    st.json({
                        "rewritten_query": r["rewritten_query"],
                        "guardrail": r["guardrail"],
                        "citation_check": r["citation_check"],
                        "timings_ms": r["timings_ms"],
                    })


prompt = st.chat_input("Ask about the filings (e.g., 'What drove revenue growth in fiscal 2023?')")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Working...", expanded=False)
        with status:
            st.write("📝 Rewriting query..." if use_rewriter else "📝 Skipping rewrite")
            st.write("🔎 Retrieving from Chroma...")
            if use_reranker:
                st.write("📐 Reranking with cross-encoder...")
            st.write("🤖 Generating grounded answer...")

        try:
            resp = answer(
                prompt,
                filters=filters or None,
                use_rewriter=use_rewriter,
                use_reranker=use_reranker,
            )
            status.update(label="Done", state="complete")

            if resp.guardrail == "out_of_scope":
                st.warning(resp.answer)
            elif resp.guardrail == "insufficient_context":
                st.info(resp.answer)
            else:
                st.markdown(resp.answer)

            resp_dict = resp.to_dict()
            cited = set(resp_dict["citation_check"]["cited_indices"])
            if resp_dict["citations"]:
                st.caption("**Sources**")
                for c in resp_dict["citations"]:
                    render_citation(c, used_in_answer=c["index"] in cited)
            if show_debug:
                with st.expander("🔍 Debug"):
                    st.json({
                        "rewritten_query": resp_dict["rewritten_query"],
                        "guardrail": resp_dict["guardrail"],
                        "citation_check": resp_dict["citation_check"],
                        "timings_ms": resp_dict["timings_ms"],
                    })

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "response": resp_dict,
            })
        except Exception as e:
            status.update(label="Error", state="error")
            st.error(f"Failed: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
