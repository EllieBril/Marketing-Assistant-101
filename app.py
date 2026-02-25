import streamlit as st
from google import genai
import wikipediaapi
import wikipedia
import time
import json
import os


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

def is_valid_industry(client, user_input):
    if user_input.replace(" ", "").isdigit():
        return False
    
    validation_prompt = f"""
    You are a strict validation gate for a Market Research tool. 
    Analyze the following input: "{user_input}"

    Rules for a "YES" verdict:
    1. The input must be in the ENGLISH language.
    2. The input must use the Latin/English alphabet (No Cyrillic, Kanji, Arabic, etc.).
    3. The input must represent a recognizable industry, sector, or business niche.
    
    Rules for a "NO" verdict:
    1. If the input is in a foreign language (e.g., 'Автомобили', '汽车').
    2. If the input is a random string of numbers or symbols.
    3. If the input is a person's name or a non-business concept.

    Answer ONLY with 'YES' or 'NO'.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=validation_prompt
        )
        verdict = response.text.strip().upper()
        return "YES" in verdict
    except Exception:
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
    
    # load from local storage if not in session state
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
                          
                            balanced_context = []
                            for text in all_texts:
                                balanced_context.append(text[:6000])
                            full_context = "\n\n NEXT SOURCE\n\n".join(balanced_context)
                            
                            prompt = f"""
                            You are a senior Market Research Analyst specializing in long-form industry reports. 
                            Your task is to write a professional report for corporate leadership based on 
                            the provided Wikipedia data for: "{industry}".
                            
                            HARD WORD COUNT LIMITS:
                            - Minimum: 450 words
                            - Maximum: 500 words
                            Your target is exactly ~475 words to stay safely within the range.
                            
                            REPORT STRUCTURE & TARGETS:
                            1. EXECUTIVE SUMMARY
                            2. MARKET DYNAMICS & SIZE
                            3. KEY TECHNOLOGICAL OR SOCIAL TRENDS
                            4. COMPETITIVE LANDSCAPE
                            5. FUTURE OUTLOOK & CHALLENGES
                            
                            STRICT INSTRUCTION: Do not exceed 500 words. If you reach the limit, conclude the sentence and stop. 
                            Use a professional, data-driven tone. Ensure the total word count is between 450 and 500.
                            
                            CONTEXT (Wikipedia Data):
                            {full_context}
                            
                            Please write the report now, ensuring it is thorough, professional, and meets the 450-500 word target.
                            """
                            try:
                                # INITIAL GENERATION
                                current_response = client.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=prompt,
                                    config={"temperature": 0.5, "top_p": 0.95, "max_output_tokens": 3000}
                                )
                                report_text = current_response.text.replace('\u0000', '').replace('\r', '').strip()
                                
                                # ITERATIVE REFINEMENT LOOP
                                import re
                                max_attempts = 3
                                for attempt in range(max_attempts):
                                    # Strip common AI chat prefixes to get accurate word count
                                    clean_report = re.sub(r'^(Here is|Certainly|Sure|As requested).*?:\n*', '', report_text, flags=re.IGNORECASE | re.DOTALL).strip()
                                    report_text = clean_report
                                    
                                    words = re.findall(r'\b\w+\b', report_text)
                                    count = len(words)
                                    
                                    if 450 <= count <= 500:
                                        break
                                    
                                    # Refine with original context and strict length instructions
                                    if count < 450:
                                        refine_instruction = f"""
                                        STRICT INSTRUCTION: The report must be 450-500 words.
                                        CURRENT COUNT: {count} words.
                                        OUTPUT FORMAT: Output ONLY the 5-section report. No intro text, no conversational filler.
                                        
                                        WIKIPEDIA DATA:
                                        {full_context}
                                        
                                        CURRENT DRAFT:
                                        {report_text}
                                        """
                                    else:
                                        refine_instruction = f"""
                                        STRICT INSTRUCTION: The report must be 450-500 words.
                                        CURRENT COUNT: {count} words.
                                        
                                        OUTPUT FORMAT: Output ONLY the 5-section report. No intro text, no conversational filler.
                                        
                                        WIKIPEDIA DATA:
                                        {full_context}
                                        
                                        CURRENT DRAFT:
                                        {report_text}
                                        """
                                    
                                    refine_response = client.models.generate_content(
                                        model="gemini-2.5-flash",
                                        contents=refine_instruction,
                                        config={"temperature": 0.5, "max_output_tokens": 4000}
                                    )
                                    report_text = refine_response.text.replace('\u0000', '').replace('\r', '').strip()

                                # FINAL DISPLAY
                                st.subheader(f"{industry} Industry Report")
                                st.write(report_text)
                                
                                final_words = re.findall(r'\b\w+\b', report_text)
                                final_count = len(final_words)
                                st.divider()
                                st.info(f"📊 Final Word Count: {final_count} words")

                                if final_count < 450 or final_count > 500:
                                    st.warning("⚠️ The AI hit its maximum refinement attempts but remained slightly out of range.")
                                    
                            except Exception as e:
                                st.error(f"Error generating report: {e}")

