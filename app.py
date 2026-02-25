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
    text_output = ""
    if response and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_output += part.text
    return text_output.replace("\u0000", "").replace("\r", "").strip()

def is_valid_industry(client, user_input):
    text = user_input.strip()
    if len(text) < 3 or text.isdigit():
        return False
    if not re.match(r'^[a-zA-Z0-9\s\&\,\.\-\/]+$', text):
        return False
    
    try:
        validation_prompt = f"Categorize the term '{text}' into one of two tags: [INDUSTRY] for real economic sectors, or [INVALID] for fictional characters, specific people, cities, or random objects. Return ONLY the tag."
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
    """Checks the floor and ceiling of the word count."""
    matches = list(re.finditer(r"\b\w+\b", text))
    count = len(matches)
    
    if count > max_words:
        cutoff_pos = matches[max_words - 1].end()
        truncated = text[:cutoff_pos].rstrip()
        last_end = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
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
        # Step 1: Validate
        with st.spinner("Step 1: Validating industry..."):
            if not is_valid_industry(client, industry):
                st.error(f"'{industry}' does not appear to be a recognized industry.")
                st.warning("Please update your inquiry with a valid economic sector to proceed.")
                st.stop()

        # Step 2: Fetch URLs
        with st.spinner("Step 2: Finding top 5 Wikipedia sources..."):
            relevant_urls, all_texts = get_wikipedia_urls(industry)

        if not relevant_urls:
            st.warning("No relevant Wikipedia pages found. Please try a broader industry name.")
            st.stop()

        st.subheader("Step 2: Relevant Wikipedia Sources")
        for i, url in enumerate(relevant_urls, 1):
            st.write(f"{i}. {url}")
        st.divider()

        # Step 3: Generate & Extend Report
        with st.spinner("Step 3: Drafting industry report (Enforcing 450-500 word limit)..."):
            full_context = "\n\n".join(text[:4000] for text in all_texts)
            
            report_prompt = (
                f"You are a Market Research Analyst.\n"
                f"Write a comprehensive industry report on '{industry}' based ONLY on the context below.\n\n"
                f"STRUCTURE:\n- Executive Summary\n- Market Dynamics\n- Key Trends\n- Competitive Landscape\n- Future Outlook\n\n"
                f"RULES:\n- Total word count MUST be strictly between 450 and 500 words.\n- Highly professional, data-driven tone.\n\n"
                f"CONTEXT:\n{full_context}"
            )

            try:
                # 1. Generate initial draft
                report_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=report_prompt,
                    config={"temperature": 0.4, "max_output_tokens": 1500}
                )
                report_text = extract_text_from_response(report_response)
                report_text = re.sub(r"^(Here is|Certainly|Sure|As requested).*?:\n*", "", report_text, flags=re.IGNORECASE | re.DOTALL).strip()
                
                # 2. The "Keep Extending" Loop
                max_extensions = 3
                for attempt in range(max_extensions):
                    current_count = word_count(report_text)
                    
                    if current_count >= 450:
                        break  # We hit the target, exit the loop!
                        
                    words_needed = 450 - current_count
                    st.toast(f"Draft is {current_count} words. Extending by ~{words_needed} words...")
                    
                    extend_prompt = (
                        f"You are a Market Research Analyst writing a report on '{industry}'.\n"
                        f"Here is your draft so far:\n\n{report_text}\n\n"
                        f"This draft is {current_count} words. It MUST be at least 450 words.\n"
                        f"Write an additional section titled 'Additional Market Insights & Risks' that is at least {words_needed + 40} words long.\n"
                        f"Use this context and DO NOT repeat what you already wrote:\n{full_context}"
                    )
                    
                    extend_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=extend_prompt,
                        config={"temperature": 0.6, "max_output_tokens": 800}
                    )
                    extension_text = extract_text_from_response(extend_response)
                    extension_text = re.sub(r"^(Here is|Certainly|Sure|As requested).*?:\n*", "", extension_text, flags=re.IGNORECASE | re.DOTALL).strip()
                    
                    report_text += "\n\n" + extension_text
                    
                # 3. Final Check & Trim
                report_text, status = enforce_word_limits(report_text, min_words=450, max_words=490)
                final_count = word_count(report_text)
                
                # 4. Display Output
                st.subheader(f"Step 3: {industry.title()} Industry Report")
                st.write(report_text)
                st.divider()
                st.info(f"📊 Final Word Count: {final_count} words")
                
                if status == "truncated":
                    st.info(f"✂️ Report trimmed to {final_count} words to strictly adhere to the sub-500 word limit.")
                elif status == "too_short":
                    st.warning(f"⚠️ Even after extending, the report is only {final_count} words. The Wikipedia context may not have enough data to support a longer report.")
                else:
                    st.success("✅ Report successfully generated and meets all constraints.")

            except Exception as e:
                st.error(f"Error generating report: {e}")








