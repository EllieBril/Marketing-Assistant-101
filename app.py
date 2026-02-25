import streamlit as st
from google import genai
import wikipediaapi
import wikipedia
import time
import json
import os
import re
import requests
from difflib import get_close_matches

# ── Industry validation helpers ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_bls_industries():
    """Fetch and cache the official BLS industry list once per session."""
    try:
        response = requests.get(
            "https://www.bls.gov/iag/tgs/iag_index_alpha.htm", timeout=10
        )
        matches = re.findall(r'<li><a href="iag[^"]+">([^<]+)</a>', response.text)
        return [m.strip().lower() for m in matches if m.strip()]
    except Exception:
        return []

def is_valid_industry(client, user_input):
    """Two-stage validation: BLS fuzzy match, then LLM fallback."""
    text = user_input.strip()

    if len(text) < 3 or text.isdigit():
        return False
    if not re.match(r'^[a-zA-Z0-9\s\&\,\.\-\/]+$', text):
        return False

    # Stage 1 — BLS fuzzy match
    bls_industries = load_bls_industries()
    if bls_industries:
        close = get_close_matches(text.lower(), bls_industries, n=1, cutoff=0.6)
        if close:
            return True 

    # Stage 2 — LLM fallback
    try:
        validation_prompt = f"""
        You are a strict classifier for a Market Research tool used by professional business analysts.
        Decide if the input below is a real, recognised business industry or economic sector.

        Input: "{text}"

        Answer YES if it is a recognised industry, sector, market, or business niche.
        Examples of YES: "SaaS", "Renewable Energy", "Cybersecurity", "Fintech", "Pet Grooming", "NFTs"

        Answer NO for everything that is not an industry:
        - Fictional characters or franchises (e.g. "Batman", "Spiderman", "Star Wars")
        - People's names (e.g. "Elon Musk", "Taylor Swift")
        - Sentences or phrases (e.g. "I am hungry", "the weather is nice")
        - Specific companies (e.g. "Apple Inc", "Tesla")
        - Standalone places (e.g. "London", "France")
        - Vague or abstract concepts (e.g. "happiness", "nature", "love")
        - Food items or consumer products (e.g. "Pizza", "Coca-Cola")

        Answer ONLY with YES or NO. No explanation.
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=validation_prompt,
            config={"temperature": 0.0}
        )
        verdict = extract_text_from_response(response).strip().upper()
        return verdict == "YES" 

    except Exception:
        return True 

# ── Core helpers ──────────────────────────────────────────────────────────────

def extract_text_from_response(response):
    """Safely extract plain text from a Gemini response object."""
    text_output = ""
    if response and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_output += part.text
    return text_output.replace("\u0000", "").replace("\r", "").strip()

def get_wikipedia_urls(industry_query):
    """Return URLs and full text for the 5 most relevant Wikipedia pages."""
    search_results = wikipedia.search(industry_query, results=5)
    wiki_api = wikipediaapi.Wikipedia(
        user_agent="MarketResearchAssistant 101",
        language="en"
    )
    urls, all_texts = [], []
    for title in search_results:
        page = wiki_api.page(title)
        if page.exists():
            urls.append(page.fullurl)
            all_texts.append(page.text)
    return urls, all_texts

def word_count(text):
    """Return the number of words in a string."""
    return len(re.findall(r"\b\w+\b", text))

def enforce_word_limits(text, min_words=450, max_words=490):
    """Enforce a word count range on the report text."""
    matches = list(re.finditer(r"\b\w+\b", text))
    count = len(matches)

    if count > max_words:
        cutoff_pos = matches[max_words - 1].end()
        truncated = text[:cutoff_pos].rstrip()
        if not truncated.endswith((".", "!", "?")):
            last_end = max(
                truncated.rfind("."),
                truncated.rfind("!"),
                truncated.rfind("?")
            )
            if last_end != -1:
                truncated = truncated[:last_end + 1]
        return truncated, "truncated"

    if count < min_words:
        return text, "too_short"

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
    if "api_key_expiry" in st.session_state:
        if current_time > st.session_state.api_key_expiry:
            st.session_state.my_api_key_persistent = ""
            st.session_state.api_key_expiry = 0
            st.session_state.api_key_saved = False
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.warning("API Key expired (30 min limit reached).")

    default_key = st.session_state.get("my_api_key_persistent", "")
    api_key_input = st.text_input(
        "Enter your API Key", type="password", value=default_key, key="api_input_field"
    )

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
industry = st.text_input("Which industry are you researching today?", key="industry_input")

if st.button("Generate Report"):
    if not industry.strip():
        st.error("Please provide an industry name to proceed.")
    elif not client:
        st.error("Please provide your API key in the sidebar.")
    else:
        # ── Step 1: Validate ────────────────
        with st.spinner("Validating industry..."):
            if not is_valid_industry(client, industry):
                st.error(
                    f'⚠️ "{industry}" does not appear to be a recognised industry. '
                    "Please enter a valid business sector or market (e.g. Renewable Energy, "
                    "Cybersecurity, Retail, Manufacturing)."
                )
                st.stop()

        # ── Step 2: Wikipedia pages ──────────────
        with st.spinner("Finding relevant Wikipedia sources..."):
            relevant_urls, all_texts = get_wikipedia_urls(industry)

        if not relevant_urls:
            st.warning("No relevant Wikipedia pages found. Try a broader industry name.")
            st.stop()

        st.subheader("Relevant Wikipedia Sources")
        for i, url in enumerate(relevant_urls, 1):
            st.write(f"{i}. {url}")
        st.divider()

        # ── Step 3: Generate report ────────────
        with st.spinner("
