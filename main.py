import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, TapasForQuestionAnswering
import torch
import warnings

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

@st.cache_resource
def load_model():
    model_name = "google/tapas-large-finetuned-wtq"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def query_model(question, df, tokenizer, model):
    # Convert all data to strings
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
    
    # Check if we got any answers
    if answers and answers[0]:
        coordinates = answers[0][0]
        if isinstance(coordinates, list) and len(coordinates) > 0:
            # Handle multiple cell answers
            answer_data = []
            for coord in coordinates:
                if isinstance(coord, tuple) and len(coord) == 2:
                    row, col = coord
                    if 0 <= row < len(df) and 0 <= col < len(df.columns):
                        answer_data.append(str(df.iloc[row, col]))
            if answer_data:
                return f"Answer: {', '.join(answer_data)}"
        elif isinstance(coordinates, tuple) and len(coordinates) == 2:
            # Handle single cell answer
            row, col = coordinates
            if 0 <= row < len(df) and 0 <= col < len(df.columns):
                return f"Answer: {df.iloc[row, col]}"
        return f"Coordinates returned by model: {coordinates}"
    else:
        return "Sorry, I couldn't find an answer to that question."

def main():
    st.set_page_config(page_title="Ask your Excel")
    st.header("Ask your Excel 📈")

    tokenizer, model = load_model()
    
    excel_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])
    if excel_file is not None:
        df = pd.read_excel(excel_file)
        st.write(df)

        user_question = st.text_input("Ask a question about your Excel: ")
        if user_question:
            with st.spinner("Processing..."):
                result = query_model(user_question, df, tokenizer, model)
                st.write(result)

if __name__ == "__main__":
    main()