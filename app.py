import streamlit as st
from google import genai
import wikipediaapi
import wikipedia
import time
import json
import os
import re
import requests

@st.cache_data(show_spinner=False)
def load_bls_industries():
    """Fetch and cache the official BLS industry list at app startup."""
    try:
        response = requests.get("https://www.bls.gov/iag/tgs/iag_index_alpha.htm", timeout=10)
        matches = re.findall(r'<li><a href="iag[^"]+">([^<]+)</a>', response.text)
        return [m.strip().lower() for m in matches if m.strip()]
    except Exception:
        return []

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_wikipedia_urls(industry_query):
    """Return URLs and full text for the 5 most relevant Wikipedia pages."""
    search_results = wikipedia.search(industry_query, results=5)
    wiki_api = wikipediaapi.Wikipedia(user_agent="MarketResearchAssistant 101", language="en")
    urls, all_texts = [], []
    for title in search_results:
        page = wiki_api.page(title)
        if page.exists():
            urls.append(page.fullurl)
            all_texts.append(page.text)
    return urls, all_texts

def extract_text_from_response(response):
    """Safely extract plain text from a Gemini response object."""
    text_output = ""
    if response and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_output += part.text
    return text_output.replace("\u0000", "").replace("\r", "").strip()

def is_valid_industry(client, user_input):
    """Strictly validate if the input is a real economic sector."""
    text = user_input.strip()
    if len(text) < 3 or text.isdigit():
        return False
    if not re.match(r'^[a-zA-Z0-9\s\&\,\.\-\/]+$', text):
        return False
    
    try:
        validation_prompt = f"Categorize the term '{text}' into one of two tags: [INDUSTRY] for real economic sectors Is this a recognized economic industry or market sector (e.g., Renewable Energy, SaaS, Automotive)?, or [INVALID] for fictional characters, specific people, cities, or random objects. Return ONLY the tag."
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=validation_prompt,
            config={"temperature": 0.0} 
        )
        verdict = extract_text_from_response(response).upper()
        return "[INDUSTRY]" in verdict
    except Exception:
        return False

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def enforce_word_limits(text, min_words=450, max_words=490):
    """Ensure text stays under the 500-word limit by truncating at the last sentence."""
    matches = list(re.finditer(r"\b\w+\b", text))
    if len(matches) > max_words:
        cutoff_pos = matches[max_words - 1].end()
        truncated = text[:cutoff_pos].rstrip()
        last_end = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_end != -1:
            truncated = truncated[:last_end + 1]
        return truncated, "truncated"
    return text, "ok"

# ── API key persistence ───────────────────────────────────────────────────────

CACHE_FILE = ".gemini_api_key.json"

def save_key_local(api_key, expiry):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"api_key": api_key, "expiry": expiry}, f)
    except Exception:
        pass

def load_key_local():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                return data.get("api_key"), data.get("expiry")
        except Exception:
            pass
    return None, None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")
    st.header("Settings")
    st.selectbox("Select LLM", ["Gemini 2.5 Flash"])

    if "my_api_key_persistent" not in st.session_state:
        saved_key, saved_expiry = load_key_local()
        if saved_key and saved_expiry:
            st.session_state.my_api_key_persistent = saved_key
            st.session_state.api_key_expiry = saved_expiry
            st.session_state.api_key_saved = True

    current_time = time.time()
    if "api_key_expiry" in st.session_state and current_time > st.session_state.api_key_expiry:
        st.session_state.my_api_key_persistent = ""
        st.session_state.api_key_expiry = 0
        st.session_state.api_key_saved = False
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.warning("API Key expired (30 min limit reached).")

    default_key = st.session_state.get("my_api_key_persistent", "")
    api_key_input = st.text_input("Enter your API Key", type="password", value=default_key)

    if st.button("Save API Key"):
        if api_key_input:
            expiry_time = time.time() + 1800
            st.session_state.my_api_key_persistent = api_key_input
            st.session_state.api_key_expiry = expiry_time
            st.session_state.api_key_saved = True
            save_key_local(api_key_input, expiry_time)
            st.success("API Key saved for 30 minutes!")
        else:
            st.error("Please enter a key before saving.")

    if not st.session_state.get("api_key_saved"):
        st.warning("Please save your API key to begin.")

# ── Gemini client ─────────────────────────────────────────────────────────────

client = None
if st.session_state.get("api_key_saved"):
    client = genai.Client(api_key=st.session_state.my_api_key_persistent)

# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("Market Research Assistant 101")
industry = st.text_input("Which industry are you researching today?")

if st.button("Generate Report"):
    if not industry.strip():
        st.error("Please provide an industry name to proceed.")
    elif not client:
        st.error("Please provide your API key in the sidebar.")
    else:
        # Step 1: Validate (Q1)
        with st.spinner("Step 1: Validating industry..."):
            if not is_valid_industry(client, industry):
                st.error(f"'{industry}' does not appear to be a recognized industry.")
                st.warning("Please update your inquiry with a valid economic sector to proceed.")
                st.stop()

        # Step 2: Fetch URLs (Q2)
        with st.spinner("Step 2: Finding top 5 Wikipedia sources..."):
            relevant_urls, all_texts = get_wikipedia_urls(industry)

        if not relevant_urls:
            st.warning("No relevant Wikipedia pages found. Please try a broader industry name.")
            st.stop()

        st.subheader("Step 2: Relevant Wikipedia Sources")
        for i, url in enumerate(relevant_urls, 1):
            st.write(f"{i}. {url}")
        st.divider()

        # Step 3: Generate Report (Q3)
        with st.spinner("Step 3: Drafting industry report (< 500 words)..."):
            full_context = "\n\n".join(text[:4000] for text in all_texts)
            
            # Formatted cleanly to avoid multiline syntax errors
            report_prompt = (
                f"You are a Market Research Analyst.\n"
                f"Write an industry report on '{industry}' based ONLY on the context below.\n\n"
                f"STRUCTURE:\n- Executive Summary\n- Market Dynamics\n- Key Trends\n- Competitive Landscape\n- Future Outlook\n\n"
                f"RULES:\n- Total word count MUST be strictly under 500 words.\n- Highly professional, data-driven tone.\n\n"
                f"CONTEXT:\n{full_context}"
            )

            try:
                report_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=report_prompt,
                    config={"temperature": 0.4, "max_output_tokens": 3000}
                )
                report_text = extract_text_from_response(report_response)

                if not report_text.strip():
                    st.error("⚠️ Model returned empty response. Please try again.")
                    st.stop()

                # Cleanup and word count enforcement
                report_text = re.sub(r"^(Here is|Certainly|Sure|As requested).*?:\n*", "", report_text, flags=re.IGNORECASE | re.DOTALL).strip()
                report_text, status = enforce_word_limits(report_text, max_words=490)
                final_count = word_count(report_text)

                # Output
                st.subheader(f"Step 3: {industry.title()} Industry Report")
                st.write(report_text)
                st.divider()
                st.info(f"📊 Final Word Count: {final_count} words")

                if status == "truncated":
                    st.info(f"✂️ Report trimmed to {final_count} words to strictly adhere to the sub-500 word limit.")
                else:
                    st.success("✅ Report successfully generated and meets all constraints.")

            except Exception as e:
                st.error(f"Error generating report: {e}")



