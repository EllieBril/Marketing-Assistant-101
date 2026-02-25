import streamlit as st
from google import genai
import wikipediaapi
import wikipedia
import time
import json
import os
import re


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def extract_text_from_response(response):
    """Safely extract plain text from a Gemini response object."""
    text_output = ""
    if response and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text_output += part.text
    return text_output.replace("\u0000", "").replace("\r", "").strip()


def is_valid_industry(client, user_input):
    """Return True if the input looks like a real industry name in English."""
    if user_input.replace(" ", "").isdigit():
        return False

    validation_prompt = f"""
    You are a strict validation gate for a Market Research tool.
    Analyze the following input: "{user_input}"

    Rules for a "YES" verdict:
    1. The input must be in the ENGLISH language.
    2. The input must use the Latin/English alphabet.
    3. The input must represent a recognizable industry, sector, or business niche.

    Rules for a "NO" verdict:
    1. If the input is in a foreign language.
    2. If the input is a random string of numbers or symbols.
    3. If the input is a person's name or a non-business concept.

    Answer ONLY with 'YES' or 'NO'.
    """
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=validation_prompt
        )
        verdict = extract_text_from_response(response).upper()
        return "YES" in verdict
    except Exception:
        return True  # Don't block the user if the API fails


def word_count(text):
    """Return the number of words in a string."""
    return len(re.findall(r"\b\w+\b", text))


def enforce_word_limits(text, min_words=450, max_words=500):
    """
    Enforce a word count range on the report text.
    - Over max_words : truncate to the last complete sentence within the limit.
    - Under min_words: return as-is with status 'too_short'.
    - Within range   : return as-is with status 'ok'.
    Returns a tuple (processed_text, status) where status is
    one of: 'ok', 'truncated', 'too_short'.
    """
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
        # Step 1 — Validate input
        with st.spinner("Validating industry..."):
            if not is_valid_industry(client, industry):
                st.error("⚠️ Invalid Input: Please enter a recognised industry name in English.")
                st.stop()

        # Step 2 — Fetch Wikipedia sources
        with st.spinner("Finding relevant Wikipedia sources..."):
            relevant_urls, all_texts = get_wikipedia_urls(industry)

        if not relevant_urls:
            st.warning("No relevant Wikipedia pages found. Try a broader industry name.")
            st.stop()

        st.subheader("Relevant Wikipedia Sources")
        for i, url in enumerate(relevant_urls, 1):
            st.write(f"{i}. {url}")
        st.divider()

        # Step 3 — Generate report section by section
        with st.spinner("Drafting your industry report..."):

            full_context = "\n\n--- NEXT SOURCE ---\n\n".join(
                text[:6000] for text in all_texts
            )

            sections = [
                "EXECUTIVE SUMMARY",
                "MARKET DYNAMICS & SIZE",
                "KEY TECHNOLOGICAL OR SOCIAL TRENDS",
                "COMPETITIVE LANDSCAPE",
                "FUTURE OUTLOOK & CHALLENGES"
            ]

            try:
                report_parts = []

                for section in sections:
                    section_prompt = f"""
                    You are a senior Market Research Analyst.
                    Write ONLY the "{section}" section of an industry report on: "{industry}".

                    STRICT RULES:
                    - Write between 90 and 100 words. No more, no less.
                    - Start directly with the section heading: {section}
                    - Write in a professional, data-driven tone.
                    - Output ONLY the section text. No commentary, no word count.

                    WIKIPEDIA CONTEXT:
                    {full_context}
                    """

                    section_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=section_prompt,
                        config={"temperature": 0.7, "top_p": 0.95, "max_output_tokens": 2000}
                    )
                    section_text = extract_text_from_response(section_response)

                    if not section_text.strip():
                        st.error(f"⚠️ Model returned empty response for section: {section}")
                        st.stop()

                    report_parts.append(section_text.strip())

                # Join all 5 sections into one report
                report_text = "\n\n".join(report_parts)

                # Strip common AI filler prefixes
                report_text = re.sub(
                    r"^(Here is|Certainly|Sure|As requested).*?:\n*",
                    "", report_text, flags=re.IGNORECASE | re.DOTALL
                ).strip()

                # Enforce word limits
                report_text, status = enforce_word_limits(report_text)
                final_count = word_count(report_text)

                # Display report
                st.subheader(f"{industry} Industry Report")
                st.write(report_text)
                st.divider()
                st.info(f"📊 Final Word Count: {final_count} words")

                if status == "too_short":
                    st.warning(f"⚠️ Report is under 450 words ({final_count} words). Try regenerating.")
                elif status == "truncated":
                    st.info(f"✂️ Report was trimmed to {final_count} words.")
                else:
                    st.success("✅ Report meets the 450–500 word target.")

            except Exception as e:
                st.error(f"Error generating report: {e}")

