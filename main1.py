# import streamlit as st

import google.generativeai as genai
import pandas as pd
from io import StringIO
import json

genai.configure(api_key="AIzaSyCLFuWZiNKwnScFFSplgAL_yhN3G2SOHzM")

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.7,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [{"category":"HARM_CATEGORY_DEROGATORY","threshold":"BLOCK_LOW_AND_ABOVE"},{"category":"HARM_CATEGORY_TOXICITY","threshold":"BLOCK_LOW_AND_ABOVE"},{"category":"HARM_CATEGORY_VIOLENCE","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_SEXUAL","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_MEDICAL","threshold":"BLOCK_MEDIUM_AND_ABOVE"},{"category":"HARM_CATEGORY_DANGEROUS","threshold":"BLOCK_MEDIUM_AND_ABOVE"}],
}

import streamlit as st

# Your text generation function (replace this with your actual implementation)
def generate_text(prompt):
    # Replace this function with your text generation logic
    return f"Generated text for prompt: {prompt}"

def generate_prompt(format_option, user_input):
    prompt = f"""input ({format_option}): {user_input}
    output:"""
    return prompt

def save_output_to_file(output, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(output)
        st.success(f"Output saved to {file_path}")
    except Exception as e:
        st.error(f"Error saving output to {file_path}: {e}")


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def main():
    st.title("Text Generation Streamlit App")

    # Sidebar for User Input
    st.sidebar.header("User Input")

    # Dropdown for Format Choice
    format_option = st.sidebar.selectbox("Choose a data format:", ["CSV", "JSON", "TXT"])

    # User Input
    user_input = st.sidebar.text_input("Enter your input:")

    # Submit Button
    if st.sidebar.button("Submit"):
        prompt = generate_prompt(format_option, user_input)
        response = genai.generate_text(
            **defaults,
            prompt=prompt
        )
        
      
        generated_result = response.result
        
        # Display the generated result in the main area
        st.header("Generated Result:")
        if format_option == "CSV":
            # Read the CSV-formatted result into a DataFrame
            df = pd.read_csv(StringIO(generated_result), sep="|", skipinitialspace=True)
            # Remove leading and trailing whitespaces in column names
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]

            st.table(df)

            csv = convert_df(df)
            
            st.sidebar.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='large_df.csv',
                mime='text/csv',
            )
        elif format_option == "JSON" :

            json_content = generated_result.replace('```', '').replace('```', '').strip()
            try:
                # Try to parse and display JSON content
                st.json(json_content, expanded=True)
            
            except json.JSONDecodeError as e:
                # If an error occurs, display the original content with st.write
                st.error(f"Error decoding JSON: {e}")
                st.write("Original content:")
                st.write(generated_result)

            st.sidebar.download_button(
                label="Download JSON",
                file_name="data.json",
                mime="application/json",
                data=generated_result,
            )
            
        elif format_option == "TXT" :
            st.write(generated_result)
            st.sidebar.download_button('Download TXT', generated_result)
            
        else:
            st.write(generated_result)
    

if __name__ == "__main__":
    main()
