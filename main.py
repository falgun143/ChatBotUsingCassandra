from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.llms import OpenAI
from langchain_community.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
import streamlit as st

ASTRA_DB_APPLICATION_TOKEN="token"#Insert your token here
ASTRA_DB_ID="Id" # Insert your id


pdfreader=PdfReader(r"F:\PdfChatBotUsingCassandraAstraDb\docs\Module 1.pdf")


from typing_extensions import Concatenate

raw_text=""

for i,page in enumerate(pdfreader.pages):
    raw_text+=page.extract_text()

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)


llm=OpenAI()
embeddings=OpenAIEmbeddings()


astra_vector_store= Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function = len,
)
  
texts=text_splitter.split_text(raw_text)


astra_vector_store.add_texts(texts)

astra_vector_index=VectorStoreIndexWrapper(vectorstore=astra_vector_store)

st.title("ChatBot")
question=st.text_input("Type the question")
button=st.button("Ask Question")

if button:
 answer=astra_vector_index.query(question,llm=llm).strip()
 st.write(answer)






