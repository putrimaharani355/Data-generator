# import streamlit as st

from google import genai
import pandas as pd
from io import StringIO
import json
import streamlit as st

client = genai.Client(api_key="AIzaSyCliDeFpsIaE8yfin9MJWSWgoV8zxMMgDE")

def generate_text(prompt):
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
        response = client.models.generate_content(
          model="gemini-2.0-flash",
          contents="give 10 data about scientist",
        )
      
        generated_result = response.text
        
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

            json_content = generated_result.replace('```json', '').replace('```', '').strip()
            try:
                # Try to parse and display JSON content
                st.json(json.loads(json_content), expanded=True)
            
            except json.JSONDecodeError as e:
                print("Invalid JSON syntax:", e)
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
