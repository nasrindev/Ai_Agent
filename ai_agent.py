import os
import requests
import textwrap
from bs4 import BeautifulSoup
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableSequence
from transformers import pipeline

# ---------------------------
# Step 1: Setup Directories
# ---------------------------
os.makedirs("inputs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------
# Step 2: Helper Functions
# ---------------------------
def load_blog(url):
    print(f"\nFetching blog: {url}")
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception as e:
        print("❌ Error fetching blog:", e)
        return ""

def load_youtube_transcript(video_id):
    print(f"\nFetching transcript for YouTube ID: {video_id}")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        print("❌ Error fetching transcript:", e)
        return ""

def load_pdf(file_path):
    import PyPDF2
    print(f"\nReading PDF: {file_path}")
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print("❌ Error reading PDF:", e)
        return ""

def load_text_file(file_path):
    print(f"\nReading text file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print("❌ Error reading text file:", e)
        return ""

# ---------------------------
# Step 3: Initialize Hugging Face Pipelines
# ---------------------------
print("\n🔹 Loading models...")

summarizer_pipeline = pipeline(
    "text2text-generation",
    model="facebook/bart-large-cnn",
    max_length=400,
    min_length=100,
    truncation=True
)

rewrite_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=300,
    truncation=True
)

summarizer_llm = HuggingFacePipeline(pipeline=summarizer_pipeline)
rewrite_llm = HuggingFacePipeline(pipeline=rewrite_pipeline)

# ---------------------------
# Step 4: Create Prompts
# ---------------------------
summary_prompt = PromptTemplate.from_template("Summarize the following text:\n\n{text}")
linkedin_prompt = PromptTemplate.from_template("Turn this summary into an engaging LinkedIn post:\n\n{text}")
tweet_prompt = PromptTemplate.from_template("Convert this summary into a short tweet thread:\n\n{text}")
script_prompt = PromptTemplate.from_template("Convert this summary into a short video script (60 seconds):\n\n{text}")

summary_chain = RunnableSequence(summary_prompt | summarizer_llm | StrOutputParser())
linkedin_chain = RunnableSequence(linkedin_prompt | rewrite_llm | StrOutputParser())
tweet_chain = RunnableSequence(tweet_prompt | rewrite_llm | StrOutputParser())
script_chain = RunnableSequence(script_prompt | rewrite_llm | StrOutputParser())

# ---------------------------
# Step 5: Main AI Agent Function
# ---------------------------
def ai_agent(source_type, source_value):
    # Step A: Load source content
    if source_type.lower() == "blog":
        content = load_blog(source_value)
    elif source_type.lower() == "youtube":
        content = load_youtube_transcript(source_value)
    elif source_type.lower() == "textfile":
        content = load_text_file(source_value)
    elif source_type.lower() == "pdf":
        content = load_pdf(source_value)
    else:
        print("❌ Invalid source type. Use: blog, youtube, textfile, pdf")
        return

    if not content:
        print("❌ No content found to process.")
        return

    # Step B: Split text into safe chunks
    print("\n🔹 Splitting content into smaller chunks...")
    chunks = textwrap.wrap(content, width=2000)
    print(f"🔹 Total chunks: {len(chunks)}")

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"   ➜ Summarizing chunk {i}/{len(chunks)}...")
        try:
            summary_part = summary_chain.invoke({"text": chunk})
            summaries.append(summary_part)
        except Exception as e:
            print(f"⚠️ Skipping chunk {i} due to error: {e}")

    combined_summary = "\n".join(summaries)

    # Step C: Generate repurposed versions
    print("\n🔹 Generating LinkedIn post...")
    linkedin_post = linkedin_chain.invoke({"text": combined_summary})

    print("🔹 Generating tweet thread...")
    tweet_thread = tweet_chain.invoke({"text": combined_summary})

    print("🔹 Generating short video script...")
    video_script = script_chain.invoke({"text": combined_summary})

    # Step D: Save all outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/ai_agent_{timestamp}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Source Type: {source_type}\nSource: {source_value}\n\n")
        f.write("=== Combined Summary ===\n" + combined_summary + "\n\n")
        f.write("=== LinkedIn Post ===\n" + linkedin_post + "\n\n")
        f.write("=== Tweet Thread ===\n" + tweet_thread + "\n\n")
        f.write("=== Video Script ===\n" + video_script)

    print(f"\n✅ Output saved successfully: {output_path}")
    print("\n✨ Summary Preview:\n", combined_summary[:500], "...")


# ---------------------------
# Step 6: User Interface (CLI)
# ---------------------------
if __name__ == "__main__":
    print("\n🤖 AI Agent Challenge")
    print("Choose input type:")
    print("1️⃣ Blog URL")
    print("2️⃣ YouTube Video ID")
    print("3️⃣ Text file from inputs/")
    print("4️⃣ PDF from inputs/")

    choice = input("Enter choice (1-4): ").strip()
    if choice == "1":
        url = input("Enter Blog URL: ").strip()
        ai_agent("blog", url)
    elif choice == "2":
        vid = input("Enter YouTube Video ID: ").strip()
        ai_agent("youtube", vid)
    elif choice == "3":
        fname = input("Enter filename in inputs/: ").strip()
        ai_agent("textfile", os.path.join("inputs", fname))
    elif choice == "4":
        fname = input("Enter PDF filename in inputs/: ").strip()
        ai_agent("pdf", os.path.join("inputs", fname))
    else:
        print("❌ Invalid choice.")
