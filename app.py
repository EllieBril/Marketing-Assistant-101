import streamlit as st
from google import genai
import wikipediaapi
import wikipedia
import time
import json
import os
import re


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def truncate_to_500(text):
    """Hard-truncate text to 500 words, ending on a complete sentence."""
    word_matches = list(re.finditer(r"\b\w+\b", text))
    if len(word_matches) <= 500:
        return text

    cutoff_pos = word_matches[499].end()
    truncated = text[:cutoff_pos].rstrip()

    # Walk back to the last sentence boundary so it ends cleanly
    if not truncated.endswith((".", "!", "?")):
        last_end = max(
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?")
        )
        if last_end != -1:
            truncated = truncated[:last_end + 1]

    return truncated


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


# ── API Key persistence ───────────────────────────────────────────────────────

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

        # Step 3 — Generate report
        with st.spinner("Drafting your industry report..."):

            full_context = "\n\n--- NEXT SOURCE ---\n\n".join(
                text[:6000] for text in all_texts
            )

            prompt = f"""
            You are a senior Market Research Analyst writing for corporate leadership.
            Write a professional industry report on: "{industry}".

            YOUR ONLY JOB: Produce EXACTLY 450-500 words. Count carefully. Do not stop early.

            MANDATORY STRUCTURE (each section must be 80-100 words):
            1. EXECUTIVE SUMMARY
            2. MARKET DYNAMICS & SIZE
            3. KEY TECHNOLOGICAL OR SOCIAL TRENDS
            4. COMPETITIVE LANDSCAPE
            5. FUTURE OUTLOOK & CHALLENGES

            Output ONLY the report. No preamble, no word count declaration, no commentary.

            WIKIPEDIA CONTEXT:
            {full_context}
            """

            try:
                # Initial generation
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"temperature": 0.5, "top_p": 0.95, "max_output_tokens": 3000}
                )
                report_text = extract_text_from_response(response)

                if not report_text.strip():
                    st.error("⚠️ The model returned an empty response. Check your API key or try again.")
                    st.stop()

                # Strip common AI filler prefixes
                report_text = re.sub(
                    r"^(Here is|Certainly|Sure|As requested).*?:\n*",
                    "", report_text, flags=re.IGNORECASE | re.DOTALL
                ).strip()

                # Iterative refinement loop
                max_attempts = 3
                for attempt in range(max_attempts):
                    count = word_count(report_text)

                    if 450 <= count <= 500:
                        break

                    if count < 450:
                        refine_instruction = f"""
                        This report is {count} words — too short.
                        Expand it to reach exactly 450-500 words.
                        Add more detail to existing sections. Do NOT add new sections.
                        Output ONLY the expanded report, no commentary.

                        REPORT TO EXPAND:
                        {report_text}
                        """
                    else:
                        refine_instruction = f"""
                        This report is {count} words — too long.
                        Trim it to exactly 450-500 words.
                        Output ONLY the trimmed report, no commentary.

                        REPORT TO TRIM:
                        {report_text}
                        """

                    refine_response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=refine_instruction,
                        config={"temperature": 0.5, "top_p": 0.95, "max_output_tokens": 3000}
                    )
                    refined = extract_text_from_response(refine_response)
                    if refined.strip():
                        report_text = refined

                # Hard truncation safety net — guarantees ≤ 500 words
                report_text = truncate_to_500(report_text)
                final_count = word_count(report_text)

                # Display
                st.subheader(f"{industry} Industry Report")
                st.write(report_text)
                st.divider()
                st.info(f"📊 Final Word Count: {final_count} words")

                if final_count < 450:
                    st.warning("⚠️ Report is under 450 words. Consider increasing refinement attempts.")
                elif final_count > 500:
                    st.warning("⚠️ Report could not be trimmed within range.")
                else:
                    st.success("✅ Report meets the 450–500 word target.")

            except Exception as e:
                st.error(f"Error generating report: {e}")




















