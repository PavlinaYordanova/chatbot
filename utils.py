from sentence_transformers import SentenceTransformer
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import streamlit as st
openai.api_key = "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"

model = SentenceTransformer('all-MiniLM-L6-v2')

embed_model = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# initialize connection to pinecone vector database
pinecone.init(api_key='b80f1498-11f9-4130-b9fb-6638d55de83b', environment='us-west1-gcp-free')

# index_name = "example_index"
# if(index_name not in pinecone.list_indexes()):
#     pinecone.create_index(index_name, dimension=1536, metric='cosine')

# connect to index in Pinecone vector database
index = pinecone.Index('test')

# view index stats
index.describe_index_stats()

def find_match(input):
    res = openai.Embedding.create(input=[input], engine=embed_model)
    xq = res['data'][0]['embedding']
    result = index.query(xq, top_k=2, include_metadata=True)

    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+ st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
