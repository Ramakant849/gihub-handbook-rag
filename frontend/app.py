import streamlit as st
from streamlit_chat import message
import requests

st.set_page_config(
    page_title="Chatbot",
    page_icon=":robot:"
)

st.header("Gitlab Handbook Buddy")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me anything about the gitlab handbook."]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hi!"]

def generate_response(query):
    url = "http://localhost:8491/api/chat"
    headers = {"Content-Type": "application/json"}
    data = {"query": query}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json().get("answer", "Error: Could not get a valid answer from the API.")
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the API: {e}"

response_container = st.container()
input_container = st.container()

with input_container:
    user_input = st.text_input("You:", key="input", on_change=lambda: st.session_state.past.append(st.session_state.input) or st.session_state.generated.append(generate_response(st.session_state.input)) or st.session_state.update(input=""))

with response_container:
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i)) 