import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0
)

# Streamlit UI
st.title("Resume Parser with LangChain & Gemini")

uploaded_file = st.file_uploader("Upload your PDF resume", type=["pdf"])

if uploaded_file is not None:
    
    # Extract text from PDF
    reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text() + "\n"
    
    st.subheader("Extracted Text (Preview):")
    st.text(resume_text[:500])  # show first 500 characters
    
    # Prompt for resume parsing
    prompt_template = """
    You are a resume parser. Extract the following fields from the resume text:
    - Name
    - Email
    - Phone
    - Education
    - Skills
    - Experience
    - Certifications
    - Projects
    - Summary
    - Achievements

    Return the output in JSON format.

    Resume Text:
    {resume_text}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Parse resume
    parsed_resume_str = chain.run({"resume_text": resume_text})
    
    # Convert JSON string to Python dict
    try:
        parsed_resume = json.loads(parsed_resume_str)
    except json.JSONDecodeError:
        st.error("Failed to parse JSON. Here is the raw output:")
        st.text(parsed_resume_str)
        parsed_resume = None
    
    if parsed_resume:
        st.subheader("Parsed Resume Fields:")
        for key, value in parsed_resume.items():
            st.markdown(f"**{key}:** {value}")
        
        # Allow user to download JSON
        st.download_button(
            label="Download Parsed Resume as JSON",
            data=json.dumps(parsed_resume, indent=4),
            file_name="parsed_resume.json",
            mime="application/json"
        )
