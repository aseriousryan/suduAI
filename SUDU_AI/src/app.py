from suduAI import SuduBotCreator
from config import *
import streamlit as st
from streamlit_chat import message

# Initialize SuduBotCreator with the parsed arguments
sudu_bot_creator = SuduBotCreator(db_path=DB_PATH, collection_name=COLLECTION_NAME, folder_path=FOLDER_PATH)

@st.cache_resource(show_spinner=True)
def create_sudu_bot():
    sudu_bot = sudu_bot_creator.create_sudu_bot()
    return sudu_bot

sudu_bot = create_sudu_bot()

def infer_sudu_bot(prompt):
    model_out = sudu_bot(prompt)
    answer = model_out['result']
    return answer

def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True, key=str(i) + "_user")
        message(history["assistant"][i],key=str(i))

def main():

    st.title("Sudu Bot 📚🤖")

    user_input = st.text_input("Enter your query")

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["I am ready to help you"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey there!"]
                
    if st.session_state :
        if st.button("Answer"):

            answer = infer_sudu_bot({'query': user_input})
            st.session_state["user"].append(user_input)
            st.session_state["assistant"].append(answer)

            if st.session_state["assistant"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()
    
    