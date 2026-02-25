import streamlit as st
from google import genai
import wikipediaapi
import wikipedia
import time
import json
import os
import re


def get_wikipedia_urls(industry_query):
    # Return the 5 most relevant pages
    search_results = wikipedia.search(industry_query, results=5)
    wiki_api = wikipediaapi.Wikipedia(
        user_agent="MarketResearchAssistant 101",
        language='en'
    )
    
    urls = []
    all_texts = []
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
    return text_output.replace('\u0000', '').replace('\r', '').strip()
    
def is_valid_industry(client, user_input):
    if user_input.replace(" ", "").isdigit():
        return False

    # 2. Define the prompt INSIDE the logic flow
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

    # 3. Call the model
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=validation_prompt
        )
        verdict = extract_text_from_response(response).upper()
        
        return "YES" in verdict
    except Exception as e:
        # If the API fails, we default to True so the user isn't blocked by a technicality
        return True


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

with st.sidebar:
    st.title("Configuration")
    st.header("Settings")

    llm_choice = st.selectbox("Select LLM", ["Gemini 2.5 Flash"]) 
    
    if "my_api_key_persistent" not in st.session_state:
        saved_key, saved_expiry = load_key_local()
        if saved_key and saved_expiry:
            st.session_state.my_api_key_persistent = saved_key
            st.session_state.api_key_expiry = saved_expiry
            st.session_state.api_key_saved = True

    # Check for expiration
    current_time = time.time()
    if "api_key_expiry" in st.session_state:
        if current_time > st.session_state.api_key_expiry:
            st.session_state.my_api_key_persistent = ""
            st.session_state.api_key_expiry = 0
            st.session_state.api_key_saved = False
            if os.path.exists(CACHE_FILE):
                os.remove(CACHE_FILE)
            st.warning("API Key expired (30 min limit reached).")

    # Use the current saved key as the default value for the input field
    default_key = st.session_state.get("my_api_key_persistent", "")
    api_key_input = st.text_input("Enter your API Key", type="password", value=default_key, key="api_input_field")
    
    if st.button("Save API Key"):
        if api_key_input:
            expiry_time = time.time() + 1800 # 30 minutes
            st.session_state.my_api_key_persistent = api_key_input
            st.session_state.api_key_expiry = expiry_time
            st.session_state.api_key_saved = True
            save_key_local(api_key_input, expiry_time)
            st.success("API Key saved for 30 minutes! It will persist even if you refresh.")
        else:
            st.error("Please enter a key before saving.")

    if not st.session_state.get("api_key_saved"):
        st.warning("Please save your API key to begin.")

# Initialize Gemini Client if saved API key is valid
client = None
if st.session_state.get("api_key_saved"):
    client = genai.Client(api_key=st.session_state.my_api_key_persistent)



st.title("Market Research Assistant 101")

industry = st.text_input("Which industry are you researching today?", key="industry_input")

if st.button("Generate Report"):
    if not industry.strip():
        st.error("Please provide an industry name to proceed.")
    elif not client:
        st.error("Please provide your API key in the sidebar.")
    else:
        #  Validation
        with st.spinner("Validating industry..."):
            if not is_valid_industry(client, industry):
                st.error("⚠️ Invalid Input: Please enter a recognized industry name in English.")
            else:
                # Finding Sources
                with st.spinner("Finding relevant Wikipedia sources..."):
                    relevant_urls, all_texts = get_wikipedia_urls(industry)
                    
                    if not relevant_urls:
                        st.warning("No relevant Wikipedia pages found. Try a broader industry name.")
                    else:
                        st.subheader("Relevant Wikipedia Sources")
                        for i, (url, text) in enumerate(zip(relevant_urls, all_texts), 1):
                            st.write(f"{i}. {url}")
                        
                        # Generating Summary
                        st.divider()
                        with st.spinner("Drafting your industry report..."):
                          
                            full_context = "\n\n NEXT SOURCE\n\n".join(
                                text[:6000] for text in all_texts
                            )
                            
                            prompt = f"""
                            You are a senior Market Research Analyst specializing in long-form industry reports. 
                            Your task is to write a professional report for corporate leadership based on 
                            the provided Wikipedia data for: "{industry}".
                            
                            HARD WORD COUNT LIMITS:
                            - Minimum: 450 words
                            - Maximum: 500 words
                
                            Use a professional, data-driven tone. Ensure the total word count is between 450 and 500.
                            
                            CONTEXT (Wikipedia Data):
                            {full_context}
                            
                            Please write the report now, ensuring it is thorough, professional, and meets the 450-500 word target.
                            """
try:
    report_text = ""  # ← add this as the very first line inside try

    # INITIAL GENERATION
    current_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.5, "top_p": 0.95, "max_output_tokens": 3000}
    )
    report_text = extract_text_from_response(current_response)

    if not report_text or not report_text.strip():
        st.error("⚠️ The model returned an empty response. Check your API key or try again.")
        st.stop()

    report_text = extract_text_from_response(current_response)

# Guard: stop early if nothing was returned
if not report_text or not report_text.strip():
    st.error("⚠️ The model returned an empty response. Check your API key or try again.")
    st.stop()                       

# ITERATIVE REFINEMENT LOOP
max_attempts = 3
for attempt in range(max_attempts):
    clean_report = re.sub(r'^(Here is|Certainly|Sure|As requested).*?:\n*', '', report_text, flags=re.IGNORECASE | re.DOTALL).strip()
    report_text = clean_report if clean_report else report_text  # don't overwrite with empty

    words = re.findall(r'\b\w+\b', report_text)
    count = len(words)

    if 450 <= count <= 500:
        break

    if count < 450:
        refine_instruction = f"""
        The following report is {count} words. Expand it to 450-500 words.
        Add detail to existing sections. No new sections. No commentary.
        Output ONLY the report.

        REPORT TO EXPAND:
        {report_text}
        """
    else:
        refine_instruction = f"""
        The following report is {count} words. Trim it to 450-500 words.
        No commentary. Output ONLY the trimmed report.

        REPORT TO TRIM:
        {report_text}
        """

    refine_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=refine_instruction,
        config={"temperature": 0.5, "top_p": 0.95, "max_output_tokens": 3000}
    )
    refined = extract_text_from_response(refine_response)
    if refined and refined.strip():  # only update if we got something back
        report_text = refined

# FINAL DISPLAY
if not report_text or not report_text.strip():
    st.error("⚠️ No report could be generated. Please try again.")
    st.stop()

words = re.findall(r'\b\w+\b', report_text)
final_count = len(words)

if final_count > 500:
    word_matches = list(re.finditer(r'\b\w+\b', report_text))
    cutoff_pos = word_matches[499].end()
    report_text = report_text[:cutoff_pos].rstrip()

    if not report_text.endswith(('.', '!', '?')):
        last_sentence_end = max(
            report_text.rfind('.'),
            report_text.rfind('!'),
            report_text.rfind('?')
        )
        if last_sentence_end != -1:
            report_text = report_text[:last_sentence_end + 1]

    final_count = len(re.findall(r'\b\w+\b', report_text))

st.subheader(f"{industry} Industry Report")
st.write(report_text)
st.divider()
st.info(f"📊 Final Word Count: {final_count} words")

if final_count < 450:
    st.warning("⚠️ Report is under 450 words. Consider increasing context or refinement attempts.")
elif final_count > 500:
    st.warning("⚠️ Report could not be trimmed within range.")
else:
    st.success(f"✅ Report meets the 450–500 word target.")



















