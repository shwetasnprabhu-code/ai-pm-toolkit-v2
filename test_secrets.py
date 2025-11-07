# test_secrets.py
import streamlit as st

st.write("OPENAI Key (first 10 chars):", st.secrets["OPENAI_API_KEY"][:10] + "...")
st.write("PINECONE Key (first 10 chars):", st.secrets["PINECONE_API_KEY"][:10] + "...")
st.write("PINECONE_ENV:", st.secrets["PINECONE_ENV"])
