from __future__ import annotations

import json
from typing import List

import streamlit as st

from src.agent import DistributorAgent
from src.config import AppConfig
from src.gemini_client import GeminiRateLimitError, GeminiServiceUnavailableError
from src.models import DistributorLead, EvidenceSnippet, LeadSearchRequest


st.set_page_config(
    page_title="Distributor Finder AI",
    page_icon="S",
    layout="wide",
)


def _render_leads(leads: List[DistributorLead]) -> None:
    if not leads:
        st.info("No distributors found yet. Run a search to generate leads.")
        return

    for idx, lead in enumerate(leads, start=1):
        with st.container(border=True):
            st.subheader(f"{idx}. {lead.company_name}")
            if lead.score_justifications:
                st.write(f"**Score:** {lead.score:.1f}/100")
            else:
                st.write("**Score:** Not scored yet")
            st.write(f"**Evidence source:** {lead.provenance}")
            st.write(f"**Country:** {lead.country}")
            st.write(f"**Website:** {lead.website or 'Not identified'}")
            st.write(f"**Summary:** {lead.summary}")
            st.write("**Why it scored this way**")
            for reason in lead.score_justifications:
                st.write(f"- {reason}")
            if lead.evidence:
                st.write("**Public evidence used**")
                for item in lead.evidence:
                    st.write(f"- {item}")
            if lead.evidence_snippets:
                st.write("**Retrieved local evidence**")
                for snippet in lead.evidence_snippets:
                    st.write(
                        f"- [{snippet.page_type}] {snippet.snippet} "
                        f"(score={snippet.retrieval_score:.3f}, source={snippet.url})"
                    )
            if lead.sources:
                st.write("**Sources**")
                for source in lead.sources:
                    st.write(f"- {source}")


def _lead_options(leads: List[DistributorLead]) -> List[str]:
    return [f"{lead.company_name} ({lead.score:.1f})" for lead in leads]


def _render_knowledge_results(results: List[EvidenceSnippet]) -> None:
    if not results:
        st.info("No local evidence matched this query yet.")
        return

    for item in results:
        with st.container(border=True):
            st.write(f"**Company:** {item.company_name}")
            st.write(f"**Page:** {item.page_title} ({item.page_type})")
            st.write(f"**Similarity:** {item.retrieval_score:.3f}")
            st.write(f"**Snippet:** {item.snippet}")
            st.write(f"**Source:** {item.url}")


def main() -> None:
    st.title("Global Distributor Finder")
    st.caption(
        "Find and rank shoe distributors by country, then generate tailored outreach and follow-up replies."
    )

    try:
        config = AppConfig.from_env()
        agent = DistributorAgent(config)
    except ValueError as exc:
        st.error(str(exc))
        st.code("Copy .env.example to .env and set GEMINI_API_KEY before running the app.")
        return

    if "leads" not in st.session_state:
        st.session_state.leads = []
    if "search_debug" not in st.session_state:
        st.session_state.search_debug = {}
    if "gemini_health" not in st.session_state:
        st.session_state.gemini_health = None
    if "knowledge_search_results" not in st.session_state:
        st.session_state.knowledge_search_results = []
    if "index_status" not in st.session_state:
        st.session_state.index_status = None

    with st.sidebar:
        st.subheader("Gemini Status")
        st.write(f"**Model:** {config.gemini_model}")
        stats = agent.get_knowledge_stats()
        st.subheader("Local Knowledge Base")
        st.write(f"**Companies:** {stats['companies']}")
        st.write(f"**Pages:** {stats['pages']}")
        st.write(f"**Chunks:** {stats['chunks']}")
        if st.button("Check Gemini Connection"):
            with st.spinner("Checking Gemini connection..."):
                st.session_state.gemini_health = agent.get_gemini_health()

        health = st.session_state.gemini_health
        if health:
            if health["ok"]:
                st.success("Gemini is connected.")
            else:
                st.error("Gemini is not responding correctly.")
            st.write(f"**Status code:** {health.get('status_code', 'Unknown')}")
            if health.get("retry_after"):
                st.write(f"**Retry-After:** {health['retry_after']}")
            st.write(f"**Response:** {health.get('message', '')}")
            if health.get("raw_body"):
                st.write("**Raw API error**")
                st.code(json.dumps(health["raw_body"], indent=2), language="json")

    discovery_tab, outreach_tab, reply_tab, knowledge_tab = st.tabs(
        ["Lead Discovery", "Outreach Drafting", "Reply Assistant", "Knowledge Base"]
    )

    with discovery_tab:
        st.subheader("Find and Score Comfortable Shoe Distributors")
        with st.form("find_and_score_distributors_form"):
            country = st.text_input("Target country", placeholder="Japan")
            desired_traits = st.text_area(
                "What you want in a distributor",
                placeholder=(
                    "Examples:\n- Existing footwear retail network\n"
                    "- Experience with comfort or walking shoe brands\n"
                    "- Strong ecommerce support\n"
                    "- Warehousing or regional logistics"
                ),
                height=160,
            )
            max_leads = st.slider("How many distributors to return", min_value=3, max_value=10, value=5)
            use_local_first = st.checkbox("Reuse local knowledge before running a fresh web search", value=True)
            submitted = st.form_submit_button("Find and score distributors")

        if submitted:
            if not country.strip():
                st.warning("Enter a target country first.")
            else:
                with st.spinner("Searching, filtering, and scoring distributors..."):
                    request = LeadSearchRequest(
                        company_name="Your footwear brand",
                        target_country=country.strip(),
                        product_focus="Comfortable shoes and comfort footwear distribution",
                        desired_traits=[
                            line.strip("- ").strip()
                            for line in desired_traits.splitlines()
                            if line.strip()
                        ],
                        max_leads=max_leads,
                    )
                    try:
                        leads = agent.find_and_rank_distributors(
                            request,
                            use_local_first=use_local_first,
                        )
                    except GeminiRateLimitError as exc:
                        st.error(str(exc))
                        leads = []
                    except GeminiServiceUnavailableError as exc:
                        st.error(str(exc))
                        leads = []
                    except Exception as exc:
                        st.error(f"Distributor discovery failed: {exc}")
                        leads = []

                    st.session_state.leads = [lead.model_dump() for lead in leads]
                    st.session_state.search_debug = agent.last_search_debug

                    if not leads:
                        st.warning("No real distributors were identified and scored from the retrieved public web results.")

        leads = [DistributorLead(**lead) for lead in st.session_state.leads]
        if leads:
            if st.button("Index current leads into local knowledge base"):
                with st.spinner("Scraping and indexing distributor pages..."):
                    index_stats = agent.index_distributor_leads(leads)
                    refreshed = agent.enrich_leads_with_local_evidence(
                        leads,
                        leads[0].request_context
                        or LeadSearchRequest(
                            company_name="Your footwear brand",
                            target_country=country.strip() or leads[0].country,
                            product_focus="Comfortable shoes and comfort footwear distribution",
                            desired_traits=[],
                            max_leads=len(leads),
                        ),
                    )
                    leads = refreshed
                    st.session_state.leads = [lead.model_dump() for lead in refreshed]
                    st.session_state.index_status = index_stats
            if st.session_state.index_status:
                stats = st.session_state.index_status
                st.success(
                    "Local indexing complete: "
                    f"{stats['companies']} companies, {stats['pages']} pages, {stats['chunks']} chunks."
                )
            st.write("**Distributors**")
            _render_leads(leads)
        debug = st.session_state.search_debug
        if debug:
            st.caption(f"Result source: {debug.get('result_source', 'unknown')}")
            with st.expander("Search Debug"):
                st.write(f"**Retrieved result count:** {debug.get('retrieved_result_count', 0)}")
                st.write(f"**Filtered distributor count:** {debug.get('filtered_count', 0)}")
                st.write(f"**Scored distributor count:** {debug.get('scored_count', 0)}")
                st.write("**Queries used**")
                for query in debug.get("queries", []):
                    st.write(f"- {query}")
                research_debug = debug.get("research_debug", {})
                if research_debug:
                    st.write("**Research engine debug**")
                    for item in research_debug.get("per_query", []):
                        st.write(
                            f"- {item.get('query', '')} | source={item.get('source', '') or 'none'} "
                            f"| results={item.get('results', 0)}"
                            + (f" | error={item.get('error')}" if item.get("error") else "")
                        )
                st.write("**Exact AI response: distributor filtering**")
                st.code(debug.get("filter_raw_response", "") or "No filtering Gemini response captured.")
                st.write("**Exact AI response: distributor scoring**")
                st.code(debug.get("score_raw_response", "") or "No scoring Gemini response captured.")

    with outreach_tab:
        st.subheader("Generate personalized outreach")
        leads = [DistributorLead(**lead) for lead in st.session_state.leads]
        if not leads:
            st.info("Run lead discovery first so the app has distributors to reference.")
        else:
            choice = st.selectbox("Choose a distributor", options=_lead_options(leads))
            selected_lead = leads[_lead_options(leads).index(choice)]
            outreach_goal = st.text_input(
                "Primary objective",
                placeholder="Secure an introductory call with the business development team",
            )
            sender_name = st.text_input("Sender name", placeholder="Jane Tan")
            sender_role = st.text_input("Sender role", placeholder="International Partnerships Lead")
            if st.button("Generate outreach"):
                with st.spinner("Drafting tailored outreach..."):
                    try:
                        message = agent.generate_outreach(
                            lead=selected_lead,
                            brand_name=(
                                selected_lead.request_context.company_name
                                if selected_lead.request_context
                                else "Your footwear brand"
                            ),
                            sender_name=sender_name.strip() or "Your Name",
                            sender_role=sender_role.strip() or "Business Development Lead",
                            outreach_goal=outreach_goal.strip()
                            or "Open a discussion about distribution partnership fit",
                        )
                    except GeminiRateLimitError as exc:
                        st.error(str(exc))
                        message = ""
                    except GeminiServiceUnavailableError as exc:
                        st.error(str(exc))
                        message = ""
                    except Exception as exc:
                        st.error(f"Outreach generation failed: {exc}")
                        message = ""
                if message:
                    if selected_lead.evidence_snippets:
                        st.write("**Retrieved company facts used**")
                        for snippet in selected_lead.evidence_snippets:
                            st.write(f"- {snippet.snippet} ({snippet.url})")
                    st.text_area("Suggested outreach", value=message, height=320)

    with reply_tab:
        st.subheader("Handle distributor replies")
        leads = [DistributorLead(**lead) for lead in st.session_state.leads]
        if not leads:
            st.info("Run lead discovery first so replies can be grounded in a selected lead.")
        else:
            choice = st.selectbox("Distributor", options=_lead_options(leads), key="reply_choice")
            selected_lead = leads[_lead_options(leads).index(choice)]
            incoming_reply = st.text_area(
                "Paste the distributor's reply",
                placeholder="Paste the email or message from the distributor here...",
                height=220,
            )
            if st.button("Generate response and action plan"):
                if not incoming_reply.strip():
                    st.warning("Paste a reply first.")
                else:
                    with st.spinner("Preparing next-step guidance..."):
                        try:
                            result = agent.respond_to_reply(
                                lead=selected_lead,
                                distributor_reply=incoming_reply,
                            )
                        except GeminiRateLimitError as exc:
                            st.error(str(exc))
                            result = None
                        except GeminiServiceUnavailableError as exc:
                            st.error(str(exc))
                            result = None
                        except Exception as exc:
                            st.error(f"Reply assistant failed: {exc}")
                            result = None
                    if result:
                        st.text_area("Suggested response", value=result["response"], height=320)
                        st.write("**Suggested action plan**")
                        for item in result["action_plan"]:
                            st.write(f"- {item}")
                        st.write("**Risk flags / notes**")
                        for item in result["notes"]:
                            st.write(f"- {item}")

    with knowledge_tab:
        st.subheader("Search Local Distributor Knowledge")
        with st.form("knowledge_search_form"):
            kb_query = st.text_input(
                "Knowledge query",
                placeholder="Find distributors in Japan that seem suitable for comfort footwear",
            )
            kb_country = st.text_input("Country filter (optional)", placeholder="Japan")
            kb_top_k = st.slider("Top chunks", min_value=3, max_value=10, value=5)
            kb_submitted = st.form_submit_button("Search local knowledge")
        if kb_submitted:
            if not kb_query.strip():
                st.warning("Enter a local knowledge query first.")
            else:
                st.session_state.knowledge_search_results = [
                    item.model_dump()
                    for item in agent.search_local_knowledge(
                        query=kb_query.strip(),
                        country=kb_country.strip(),
                        top_k=kb_top_k,
                    )
                ]
        _render_knowledge_results(
            [EvidenceSnippet(**item) for item in st.session_state.knowledge_search_results]
        )

if __name__ == "__main__":
    main()
