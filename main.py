import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from transformers import AutoTokenizer, TapasForQuestionAnswering
import warnings
import app
import app2

# Suppress FutureWarnings
warnings.filterwarnings("ignore")

@st.cache_resource
def load_model():
    model_name = "google/tapas-large-finetuned-wtq"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def query_model(question, df, tokenizer, model):
    table = df.astype(str)
    
    inputs = tokenizer(table=table, queries=[question], padding='max_length', return_tensors="pt")
    outputs = model(**inputs)
    
    predicted_answer_coordinates = outputs.logits.detach().numpy()
    
    loss = outputs.loss.detach() if outputs.loss is not None else None
    
    answers = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        loss
    )
    
    if answers and answers[0]:
        coordinates = answers[0][0]
        if isinstance(coordinates, list) and len(coordinates) > 0:
            answer_data = []
            for coord in coordinates:
                if isinstance(coord, tuple) and len(coord) == 2:
                    row, col = coord
                    if 0 <= row < len(df) and 0 <= col < len(df.columns):
                        answer_data.append(str(df.iloc[row, col]))
            if answer_data:
                return f"Answer: {', '.join(answer_data)}"
        elif isinstance(coordinates, tuple) and len(coordinates) == 2:
            row, col = coordinates
            if 0 <= row < len(df) and 0 <= col < len(df.columns):
                return f"Answer: {df.iloc[row, col]}"
        return f"Coordinates returned by model: {coordinates}"
    else:
        return "Sorry, I couldn't find an answer to that question."

# Function to render the Excel QA page
def render_excel_qa():
    st.header("Ask your Excel 📈")

    tokenizer, model = load_model()
    
    # Load the default "student_academics" file
    df = pd.read_excel("student_academics.xlsx")    

    user_question = st.text_input("Ask a question about the student academics data: ")
    if user_question:
        with st.spinner("Processing..."):
            result = query_model(user_question, df, tokenizer, model)
            st.write(result)

# Main function to render the app
def main():
    st.set_page_config(page_title="Excel & Academic Support Chatbot")
    
    with st.sidebar:
        st.title("Academic Support Service")
        st.write(" Choose an option from the menu below for our services:")

        selected = option_menu('Your Academic Support',
                               [
                                'Your Statistics',
                                'Excel QA',
                                'Academic Support Chatbot',
                               ],
                                default_index=0)

    if (selected == 'Your Statistics'):
        app2.main()
    if (selected == 'Excel QA'):
        render_excel_qa()
    if (selected == 'Academic Support Chatbot'):
        app.main()

if __name__ == "__main__":
    main()
