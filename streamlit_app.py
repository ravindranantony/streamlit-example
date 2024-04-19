import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Set the title of the web app
st.title('Tamil Knowledge Center')

# Text box for user input
user_input = st.text_input("Ask me anything in Tamil:")

def get_response(text):
    # Encode the user input and add the necessary tokens
    inputs = tokenizer.encode(text + tokenizer.eos_token, return_tensors="pt")

    # Generate a response
    output = model.generate(inputs, max_length=512, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Handling user interaction
if user_input:
    response = get_response(user_input)
    st.write(response)

# Run the Streamlit app (this command is only necessary when running the script manually, not needed in the script itself)
# st.run()
