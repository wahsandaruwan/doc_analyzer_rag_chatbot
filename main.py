import streamlit as st
import tempfile
import os
from rag import Rag

def display_messages():
  for message in st.session_state.messages:
    with st.chat_message(message['role']):
      st.markdown(message['content'])

def process_file():
  st.session_state.messages = []

  if st.session_state["file_uploader"]:
    for file in st.session_state["file_uploader"]:
      with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(file.getbuffer())
        file_path = tf.name

      st.session_state["assistant"].feed(file_path)
      os.remove(file_path)

  # Removed st.rerun() call

def process_input():
  if prompt := st.chat_input("What can i do?"):
    with st.chat_message("user"):
      st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state["assistant"].ask(prompt)
    with st.chat_message("assistant"):
      st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

def main():
  st.title("OUSL Chatbot")

  if "assistant" not in st.session_state:
    st.session_state["assistant"] = Rag()
  if "messages" not in st.session_state:
    st.session_state.messages = []

  st.file_uploader(
    "Upload the document",
    type=["pdf"],
    key="file_uploader",
    on_change=process_file,
    label_visibility="collapsed",
    accept_multiple_files=True,
  )

  display_messages()
  process_input()

if __name__ == "__main__":
  main()