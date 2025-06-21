import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, Runner
import PyPDF2

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("🔐 Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Gemini-compatible client
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

# Page settings
st.set_page_config(page_title="🔬 AI Research Agent", page_icon="🧠", layout="wide")
st.title("🧠 AI Research Agent")
st.markdown("Empowering your research journey with Gemini 2.0 Flash 🚀")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    mode = st.radio("🧠 Response Style", ["📝 Simple", "🧒 Explain Like I'm 5", "💻 Technical"])
    st.markdown("🔄 Powered by Gemini 2.0 Flash")
    st.divider()
    st.caption("📌 Pro tip: Use clear prompts for best results.")

instructions = {
    "📝 Simple": "Give a short, clear, and factual response for any academic or research question.",
    "🧒 Explain Like I'm 5": "Explain the answer in very simple, easy-to-understand terms, like to a 5-year-old.",
    "💻 Technical": "Provide a detailed and technical explanation with examples, references, or formulas if needed."
}

# Agent with selected mode
agent = Agent(
    name="Research Agent",
    instructions=instructions[mode]
)

# Input section
st.subheader("🔍 Ask a Research Question")
question = st.text_input("💬 What do you want to learn about?", placeholder="e.g., What is quantum entanglement?")

# Answer generation
if st.button("🚀 Get Answer") and question.strip():
    with st.spinner("Thinking with Gemini... 🤖"):
        response = asyncio.run(Runner.run(agent, input=question, run_config=config))
    st.success("✅ Response Ready")
    st.markdown("### 📄 Result")
    st.write(response.final_output)

# --- Additional Tools ---
st.subheader("🛠️ Extra Research Tools")
tool = st.selectbox("Choose a tool", [
    "None",
    "📄 PDF Summarization",
    "🧠 Keyword Extraction",
    "📝 APA Reference Generator",
    "💡 Concept Explainer"
])

if tool == "📄 PDF Summarization":
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf and st.button("📘 Summarize PDF"):
        reader = PyPDF2.PdfReader(pdf)
        text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])[:8000]
        prompt = f"Summarize the following research paper:\n\n{text}"
        with st.spinner("Summarizing PDF..."):
            response = asyncio.run(Runner.run(agent, input=prompt, run_config=config))
        st.success("✅ Summary:")
        st.write(response.final_output)

elif tool == "🧠 Keyword Extraction":
    content = st.text_area("Paste content for keyword extraction")
    if st.button("🔑 Extract Keywords") and content:
        prompt = f"Extract the most relevant keywords from this content:\n\n{content}"
        with st.spinner("Extracting keywords..."):
            response = asyncio.run(Runner.run(agent, input=prompt, run_config=config))
        st.success("✅ Keywords:")
        st.write(response.final_output)

elif tool == "📝 APA Reference Generator":
    reference_text = st.text_area("Paste a paragraph or citation text")
    if st.button("📄 Generate APA Reference") and reference_text:
        prompt = f"Generate APA references from the following text:\n\n{reference_text}"
        with st.spinner("Generating reference..."):
            response = asyncio.run(Runner.run(agent, input=prompt, run_config=config))
        st.success("✅ APA References:")
        st.write(response.final_output)

elif tool == "💡 Concept Explainer":
    concept = st.text_input("Enter concept to explain")
    if st.button("🧠 Explain Concept") and concept:
        prompt = f"Explain the following concept in simple academic terms:\n\n{concept}"
        with st.spinner("Explaining concept..."):
            response = asyncio.run(Runner.run(agent, input=prompt, run_config=config))
        st.success("✅ Explanation:")
        st.write(response.final_output)

# Footer
st.markdown("""<hr style=\"margin-top: 3rem; margin-bottom: 1rem;\">""", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; font-size: 16px; color: black;'>
        🛠️ Created by <strong>Rahat Bano💖</strong> |
        <a href='https://github.com/rahatbano58' target='_blank'>🔗 GitHub: rahatbano58</a><br>
        © 2025 All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
