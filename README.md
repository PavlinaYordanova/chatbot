streamlit
streamlit_chat
langchain
sentence_transformers
openai
pinecone

"""In this project, we take you through the process of creating a chatbot that leverages the power of Langchain, OpenAI's ChatGPT, Pinecone, and Streamlit Chat. This chatbot not only responds to user queries but also refines queries and pulls information from its own document index, making it incredibly efficient and capable of answering in-depth questions."""

===============================================================
============Pinecone First Account with Free Vector Database ==
user: p.yankovainnovasys@gmail.com
password: &Yank0v@2023
Default Project API Keys: 6c4992db-2ce5-46ab-8d78-d3a5603fbef0
pinecone environment: us-west4-gcp-free
Default Project Indexes: example-index
===============================================================
=========Pinecone Second Account with Free Vector Database ====
username: p.yordanovainnovasys@gmail.com
password: p.y0rd@nov@12
Default Project API Keys: b80f1498-11f9-4130-b9fb-6638d55de83b
pinecone environment: us-west1-gcp-free
Default Project Indexes: test
===============================================================




 # Colab.research.google.com

!pip install --upgrade langchain openai -q
!pip install sentence_transformers -q

!pip install unstructured -q
!pip install unstructured[local-inference] -q
!pip install detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2 -q
!apt-get install poppler-utils
!mkdir new_folder
!pip install pypdf
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/Общи инструкции за работа с Бързи връзки.pdf")
pages = loader.load_and_split()
pages[0]
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(pages,chunk_size=1000,chunk_overlap=200):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(pages)
  return docs

docs = split_docs(pages)
print(len(docs))
print(docs[1].page_content)
!pip install tiktoken -q
!pip install pinecone-client
import os
import getpass
PINECONE_API_KEY = getpass.getpass('Pinecone API Key:')
PINECONE_ENV = getpass.getpass('Pinecone Environment:')
os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
!pip install openai
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()



# Uploading our data to Pinecone vector database

import pinecone
from langchain.vectorstores import Pinecone

# initialize pinecone
pinecone.init(
    api_key="6c4992db-2ce5-46ab-8d78-d3a5603fbef0",  # find at app.pinecone.io
    environment="us-west4-gcp-free"  # next to api key in console
)

index_name = "example-index"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


# Making queries from our data stored in pinecone vector database

def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

query = "Как се прави изтриване на меню от модул „Бързи връзки“"
similar_docs = get_similiar_docs(query)
similar_docs