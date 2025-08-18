# initial imports
# own experiment

import streamlit as st
from pathlib import Path
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytube import YouTube


# steamlit setup
st.set_page_config(page_title="YouTube Video Summarizer", page_icon="&", layout="wide")
st.title("YouTube Video Summarizer with Groq & LangChain")

yt = YouTube(url, use_oauth=True, allow_oauth_cache=True) # Add these parameters
loader = YoutubeLoader(video_info=yt.vid_info)

# radio_opt = ["Summarize youtube video", "Summarize website"]
# selected_option = st.sidebar.radio(label = "Choose to summarize", options=radio_opt)


api_key = st.sidebar.text_input(label="Groq api key", type="password")


# llm model
llm = ChatGroq(groq_api_key = api_key, model_name = "llama3-8b-8192", streaming=True)

# --- Define Prompt Template for Summarization ---
# It's crucial that your prompt includes a '{context}' variable
# where the loaded transcript will be "stuffed".
summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer. Summarize the following YouTube video transcript concisely and accurately. Focus on key points and main ideas. Aim for 300 words or less. Transcript:\n\n{context}"),
    ("human", "Please summarize this YouTube video transcript."),
])


# --- Create the Summarization Chain ---
# This chain takes documents (transcript chunks) and stuffs them into the prompt
summarize_stuff_chain = create_stuff_documents_chain(llm, summarization_prompt)

# --- Streamlit Application Logic ---
youtube_url = st.text_input("Enter YouTube Video URL:", "")

if youtube_url:
    st.subheader("Processing Video...")

    try:
        # 1. Load the YouTube transcript
        # Ensure youtube_transcript_api is installed: pip install youtube-transcript-api
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
        print(youtube_url)
        docs = loader.load() # This returns a list of Document objects with the transcript

        if not docs:
            st.error("Could not load transcript for this video. It might not have one or be unavailable.")
            st.stop()

        full_transcript = docs[0].page_content # Get the full transcript text

        # 2. Split the transcript into manageable chunks (if needed for very long videos)
        # While create_stuff_documents_chain can handle a single large doc if it fits,
        # splitting is good practice for robust summarization of long videos.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        transcript_chunks = text_splitter.split_documents(docs)

        st.info(f"Loaded transcript with {len(transcript_chunks)} chunks.")

        # 3. Summarize the transcript
        with st.spinner("Generating summary..."):
            # StreamlitCallbackHandler can be used here for more detailed step-by-step output
            # but for simple summarization, st.write is often sufficient.
            # If you want to see agent thoughts, you might wrap this in a temp chain with callbacks
            # and potentially use create_retrieval_chain or an AgentExecutor if context is dynamic.

            # Simple summarization:
            # The create_stuff_documents_chain expects a list of Document objects for 'context'
            summary_response = summarize_stuff_chain.invoke({"context": transcript_chunks})

            st.subheader("Summary:")
            st.write(summary_response) # The actual summary content
            st.success("Summary generated successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check the YouTube URL and ensure the video has an English transcript.")
