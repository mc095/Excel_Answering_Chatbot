import streamlit as st
from huggingface_hub import InferenceClient
from langchain.prompts import PromptTemplate

client = InferenceClient(
    "microsoft/Phi-3-mini-4k-instruct",
    token="--include your hf token key here--"
)

SYSTEM_PROMPT_ACADEMIC = """You are an academic support chatbot. 
Provide detailed and supportive advice to help students improve their performance in various subjects. 
Answer the following question in a helpful and encouraging manner, focusing on specific topics to prepare for competitive exams. Generate only 6 points. Do not generate any text after 'Good luck!'
End your total answer with 'Good luck!'"""

prompt_template = PromptTemplate(
    input_variables=["system_prompt", "subject", "question"],
    template="{system_prompt}\n\nSubject: {subject}\nQuestion: {question}\nAssistant:"
)

def main():
    # Streamlit app layout
    st.title("Academic Support Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm your academic support chatbot. How can I help you improve your studies today?"}
        ]

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input fields
    subject = st.text_input("Enter the subject you need help with:")
    user_question = st.chat_input("Ask a question about improving your score:")

    if subject and user_question:
        # Append user message to the session state
        user_message = f"Subject: {subject}\nQuestion: {user_question}"
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.chat_message("user").write(user_message)

        # Format prompt using LangChain's PromptTemplate
        formatted_prompt = prompt_template.format(
            system_prompt=SYSTEM_PROMPT_ACADEMIC,
            subject=subject,
            question=user_question
        )

        response = client.text_generation(
            formatted_prompt,
            max_new_tokens=450,  # Increased token limit
            stream=False,
        )

        # Post-processing: Ensure response ends exactly with 'Good luck!' and remove any additional content
        response = response.split("Good luck!")[0].strip() + " Good luck!"

        # Append assistant message to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
