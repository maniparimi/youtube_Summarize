## Used Pytube here, not working giving 404 error. Fixed by using yt-dlp in the ytdlp2 file


import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import UnstructuredURLLoader


st.set_page_config(page_title="YouTube/Website Summarizer")
st.title("Summarize Content from YouTube or Website")
st.subheader("Summarize url:")


api_key = st.sidebar.text_input('Grok API Key', value='', type='password')

url = st.text_input('Enter URL (YouTube or Website)', label_visibility='collapsed')


llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)
prompt_template = "Provide a summary of the following content in 300 words: {text}"
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


if st.button('Summarize Content from YouTube or Website'):
    if not api_key.strip() or not url.strip():
        st.error('Please provide the information.')
    elif not validators.url(url):
        st.error('Please enter a valid URL. It can be a YouTube video URL or website URL.')
    else:
        # Proceed with summarization
        try:
            with st.spinner('Waiting...'):
                if 'youtube.com' in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers=headers)
                docs = loader.load()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
        except Exception as e:
            st.exception(e)




