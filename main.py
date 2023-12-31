from langchain.chat_models import ChatOpenAI
from dotenv.main import load_dotenv
import os
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
load_dotenv()
openai_key = os.environ['OPENAI_API_KEY']

import streamlit as st
from streamlit_chat import message  # a Python package that provides a chatbot interface for Streamlit applications
from utils import *

load_dotenv()
openai_key = os.environ['OPENAI_API_KEY']
# headers = {
#     "authorization": st.secrets["OPENAI_API_KEY", "PINECONE_API_KEY"],
#     "content-type": "application/json"
# }

st.title("МОНЕТА Chatbot :books:") # giving a cool name for our Chatbot using st.title() function
st.markdown(" Powered by 🦜 LangChain + 🧠 OpenAI + 🚀 Pinecone + 💬 Streamlit")

# initialize the chatbot by giving it a starter message at the first app run:
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Как мога да Ви помогна?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)

# creating a ConversationBufferWindowMemory object with k=3 and assigns it to the session state variable 'buffer_memory' 
# ConversationBufferWindowMemory keeps the latest pieces of the conversation in raw form

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    # chat_memory=ChatMessageHistory(messages=[]) output_key=None input_key=None return_messages=True human_prefix='Human' ai_prefix='AI' memory_key='history' k=3

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""
    Ти си опитен специалист в поддръжка на ERP система Монета. 
    Твоята задача е да помагаш на потребителите с ежедневната им работа, като отговаряш на въпроси от ралично естество, 
    свързани с функционалности в Монета. 
    Ако не разбираш правилно въпроса, който задава потребителя, моля поискай допълнителни пояснения, за да си сигурен,
    че отговорите са максимално точни. 
    Ти трябва да отговориш на въпроса, запитването или оплакването възможно най-вярно, и да обясниш всичко стъпка по стъпка
    като на 8 годишно дете, използвайки предоставения текст.
    Aко отговорът не се съдържа в този текст, кажи 'Въпросът не е свързан с програмата.'"""
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
input_container = st.container()
clear_text = st.container()

with input_container:
    # creating a text area for the user input query using st.text_input() function
    query = st.text_input("Въпрос: ", key="input", placeholder="МОНЕТА отговаря! Питайте ме ...") 
    if query:
        with st.spinner("Мисля..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            context = find_match(refined_query)
            # # print(context)
             
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with clear_text:
    def clear_text():
        st.session_state["input"] = ""
    
    st.button("Clear Text", on_click=clear_text)


# Displaying the response. If there are any generated responses in the st.session_state object, a for loop is initiated. 
# For each generated response, the message() function is called twice to display the query made by the user and the response generated by the llm
# The key parameter is used to uniquely identify each message.
with response_container:
    if st.session_state['responses']: # ['Как мога да Ви помогна?']

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i), avatar_style='personas')
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
                

  
                

                

  
