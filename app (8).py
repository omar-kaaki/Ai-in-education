import streamlit as st
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the fine-tuned GPT-2 model and tokenizer
model_path = "./fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Load the JSON dataset
json_dataset_path = "further_expanded_educational_chatbot_dataset.json"

with open("/content/further_expanded_educational_chatbot_dataset (1).json", "r") as file:
    json_data = json.load(file)

# Convert JSON data into a DataFrame
json_intents = json_data["intents"]
json_dataset = pd.DataFrame([
    {"StudentQuery": pattern, "Response": intent["responses"][0]}
    for intent in json_intents
    for pattern in intent["patterns"]
])

# Preprocess the data for text recommendations
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(json_dataset['StudentQuery'].fillna(""))

# Initialize chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize user input state if it doesn't exist
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def is_study_related(query):
    """
    Determine if a query is related to studies using keywords.
    """
    study_keywords = ["study", "exam", "subject", "syllabus", "learning", "school", "university", "test", "education"]
    return any(keyword in query.lower() for keyword in study_keywords)

def get_dataset_response(query):
    """
    Find the best matching response from the dataset using cosine similarity.
    """
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_match_idx = similarities.argmax()
    if similarities[best_match_idx] > 0.3:  # Adjust threshold as needed
        return json_dataset.iloc[best_match_idx]['Response']
    return None

# Streamlit App Configuration
st.title("Student Chatbot")
st.write("Hi! I'm your academic assistant. Feel free to ask me anything or request study help.")

# Display chat history dynamically
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Chatbot:** {message['content']}")

# User Input
user_input = st.text_input("Your Query:", value=st.session_state.user_input, key="input_box")

if st.button("Send"):
    if user_input.strip() == "":
        st.warning("Please type something to start a conversation!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Determine response
        if is_study_related(user_input):
            # Fetch related dataset response
            dataset_response = get_dataset_response(user_input)
            if dataset_response:
                response = f"Based on your question, I recommend: {dataset_response}. Can I assist you further?"
            else:
                response = "I couldn't find specific advice in my notes, but I'd suggest focusing on your key topics and practicing regularly."
        else:
            # Generate general conversational response using GPT-2
            input_text = f"Student: {user_input}\nAssistant:"
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            output = model.generate(
                input_ids,
                max_new_tokens=50,  # Limit the response length
                num_return_sequences=1,
                temperature=0.8,  # Add randomness to avoid repetition
                top_p=0.9,  # Nucleus sampling for diverse responses
                top_k=50,  # Consider top 50 tokens
                repetition_penalty=1.2,  # Penalize repetition
                do_sample=True,
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.split("Assistant:")[-1].strip()

        # Add chatbot response to chat history
        st.session_state.messages.append({"role": "bot", "content": response})

        # Clear the user input after sending
        st.session_state.user_input = ""
