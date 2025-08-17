import streamlit as st
import subprocess
import os
import tempfile
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import sys # Import sys to check the operating system


st.set_page_config(page_title="YouTube/Website Summarizer")
st.title("Summarize Content from YouTube or Website")
st.subheader("Enter URL:")


# Input fields for API Key and URL
GROQ_API_KEY = st.sidebar.text_input('Groq API Key', value='', type='password')
url = st.text_input('Enter URL (YouTube or Website)', label_visibility='collapsed')


# Summarize button
if st.button('Summarize Content'):
    if not GROQ_API_KEY.strip() or not url.strip():
        st.error('Please provide both the Groq API Key and the URL.')
    elif not validators.url(url):
        st.error('Please enter a valid URL. It can be a YouTube video URL or website URL.')
    else:
        # Proceed with summarization
        try:
            with st.spinner('Waiting...'):
                docs = []

                if 'youtube.com' in url:
                    # --- Using yt-dlp to extract transcript ---
                    with tempfile.TemporaryDirectory() as tmpdir:
                        output_template = os.path.join(tmpdir, '%(title)s.%(ext)s')
                        
                        # Dynamically determine yt-dlp executable path based on OS
                        yt_dlp_executable = "yt-dlp" # Default to rely on PATH
                        if sys.platform.startswith('linux') or sys.platform == 'darwin': # Linux or macOS
                            # Try common install path first
                            if os.path.exists("/usr/local/bin/yt-dlp"):
                                yt_dlp_executable = "/usr/local/bin/yt-dlp"
                            elif os.path.exists(os.path.expanduser("~/.local/bin/yt-dlp")):
                                yt_dlp_executable = os.path.expanduser("~/.local/bin/yt-dlp")
                            # If using a virtual environment, it might be in venv/bin
                            elif os.path.exists(os.path.join(sys.prefix, 'bin', 'yt-dlp')):
                                yt_dlp_executable = os.path.join(sys.prefix, 'bin', 'yt-dlp')
                        elif sys.platform == 'win32': # Windows
                            # If yt-dlp.exe is in the same directory as the script
                            if os.path.exists("yt-dlp.exe"):
                                yt_dlp_executable = "yt-dlp.exe"
                            # If yt-dlp.exe is in Python's Scripts folder
                            elif os.path.exists(os.path.join(sys.exec_prefix, 'Scripts', 'yt-dlp.exe')):
                                yt_dlp_executable = os.path.join(sys.exec_prefix, 'Scripts', 'yt-dlp.exe')
                            # Default to rely on PATH (might work if set)
                            else:
                                yt_dlp_executable = "yt-dlp.exe" # Still need .exe on Windows

                        cmd = [
                            yt_dlp_executable, # Use the determined executable path
                            '--skip-download',  
                            '--write-auto-subs',
                            '--sub-langs', 'en', 
                            '--output', output_template,
                            url
                        ]
                        
                        try:
                            # Using capture_output=True for more informative error messages
                            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        except subprocess.CalledProcessError as e:
                            st.error(f"yt-dlp failed with return code {e.returncode}: {e.stderr}")
                            st.exception(e) 
                            st.warning("Failed to extract transcript with yt-dlp. Make sure yt-dlp is installed, executable, and the video has captions.")
                            st.stop() 
                        except FileNotFoundError:
                            st.error(f"yt-dlp executable not found. Please verify the installation and the path '{yt_dlp_executable}'.")
                            st.info("Try running `which yt-dlp` (Linux/macOS) or `where yt-dlp` (Windows) in your terminal to find its location.")
                            st.stop()
                        
                        transcript_content = ""
                        for filename in os.listdir(tmpdir):
                            if filename.endswith(('.vtt', '.json', '.srv3', '.ass', '.srt')): 
                                with open(os.path.join(tmpdir, filename), 'r', encoding='utf-8') as f:
                                    transcript_content = f.read()
                                break 
                        
                        if transcript_content:
                            parsed_transcript = []
                            for line in transcript_content.split('\n'):
                                if not (line.strip().isdigit() or '-->' in line or line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:')):
                                    if line.strip():
                                        parsed_transcript.append(line.strip())
                            final_transcript = " ".join(parsed_transcript)

                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=3000, 
                                chunk_overlap=300, 
                                separators=["\n\n", "\n", " ", ""] 
                            )
                            docs = text_splitter.create_documents([final_transcript]) 
                        else:
                            st.error("No transcript file found by yt-dlp after extraction. The video may not have captions available.")
                            st.stop()

                else: 
                    headers = {"User-Agent": "Mozilla/50 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers=headers)
                    
                    web_docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=3000, 
                        chunk_overlap=300,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    docs = text_splitter.split_documents(web_docs)


                llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
                
                map_prompt_template = """The following is a part of a larger document. Summarize this section to extract the most important information:
                {text}
                Summary:
                """
                map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

                reduce_prompt_template = """You are an expert summarizer. You have been given several summaries of a document. 
                Combine these summaries into a single, comprehensive summary of approximately 300 words. 
                Ensure the final summary covers all key aspects and is well-organized.

                Here are the individual summaries:
                {text}

                Final Summary:
                """
                reduce_prompt = PromptTemplate(template=reduce_prompt_template, input_variables=["text"])

                chain = load_summarize_chain(
                    llm, 
                    chain_type="map_reduce", 
                    map_prompt=map_prompt, 
                    combine_prompt=reduce_prompt, 
                    verbose=False # Set to True for debugging, False for cleaner UI
                )
                
                output_summary = chain.invoke(docs)
                st.success("--- Summary ---")
                st.write(output_summary['output_text'])

        except Exception as e:
            st.exception(e) 
            st.error("An unexpected error occurred during summarization.")
