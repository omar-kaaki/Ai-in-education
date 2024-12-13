import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import requests
import hashlib
import json
import io
from datetime import datetime
import os

# Constants
USERS_FILE = 'users.json'
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# -----------------
# Utility Functions
# -----------------

def save_user_history(username, answers, cluster, recommendation):
    history = load_users()
    if not history:
        history = {"users": []}

    # Ensure users list exists
    if "users" not in history:
        history["users"] = []

    # Find or create user entry
    user_entry = next((user for user in history["users"] if user["username"] == username), None)
    if not user_entry:
        user_entry = {"username": username, "history": []}
        history["users"].append(user_entry)

    # Ensure history key exists in user_entry
    if "history" not in user_entry:
        user_entry["history"] = []

    # Append new history
    user_entry["history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "answers": answers,
        "cluster": cluster,
        "recommendation": recommendation
    })

    save_users(history)


def generate_history_file(username):
    history = load_users()
    user_entry = next((user for user in history["users"] if user["username"] == username), None)
    if user_entry and "history" in user_entry:
        history_content = user_entry["history"]
        history_df = pd.DataFrame(history_content)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            history_df.to_excel(writer, index=False, sheet_name='History')
        output.seek(0)
        return output
    else:
        return io.BytesIO("No history available for this user.".encode("utf-8"))


def load_users():
    try:
        if not os.path.exists(USERS_FILE):
            return {"users": []}
        with open(USERS_FILE, "r") as f:
            content = f.read().strip()
            return json.loads(content) if content else {"users": []}
    except (json.JSONDecodeError, FileNotFoundError):
        return {"users": []}

def save_users(data):
    def convert_to_native(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        return obj

    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=4, default=convert_to_native)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    users_data = load_users()
    for user in users_data['users']:
        if user['username'] == username:
            return False  # Username already exists
    users_data['users'].append({"username": username, "password": hash_password(password)})
    save_users(users_data)
    return True

def authenticate_user(username, password):
    users_data = load_users()
    for user in users_data['users']:
        if user['username'] == username and user['password'] == hash_password(password):
            return True
    return False

def encode_categorical(value, options, unknown_value=0.5):
    le = LabelEncoder()
    le.fit(options)
    if value in le.classes_:
        return le.transform([value])[0] / (len(options) - 1)
    return unknown_value

def generate_expanded_recommendation(base_recommendation, cluster_id):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        
    f"Based on the user's learning style and preferences, "
    f"here is the recommendation: {base_recommendation}\n\n"
    "Expand this recommendation with detailed suggestions and provide some useful online resources tailored to the user's needs.( please provide the link to the resource and address the user by first person perspective)"


    )

    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return "Could not generate recommendation due to an API error."

# -----------------
# Load Models
# -----------------
clustering_nn = load_model('490-FINAL\models\clustering_nn.h5', compile=False)
kmeans_model = joblib.load('490-FINAL\models\kmeans_model.pkl')

try:
    scaler = joblib.load('490-FINAL\models\scaler.pkl')
except FileNotFoundError:
    st.warning("Scaler not found. A new scaler will be created if necessary.")
    scaler = None

clustering_encoder = Model(
    inputs=clustering_nn.input,
    outputs=clustering_nn.get_layer('latent_layer').output
)

# -----------------
# Streamlit GUI
# -----------------
st.title("Personalized Learning Recommendation System")
page = st.sidebar.selectbox("Navigation", ["Login", "Sign Up", "Survey"])

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if page == "Sign Up":
    st.subheader("Create a New Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Sign Up"):
        if register_user(username, password):
            st.success("Account created successfully! Please log in.")
        else:
            st.error("Username already exists. Try another one.")

elif page == "Login":
    st.subheader("Log In to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        if authenticate_user(username, password):
            st.success("Login successful!")
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            # Reset chat history for the new user
            st.session_state["chat_history"] = []
            st.session_state["file_content"] = None
        else:
            st.error("Invalid username or password.")
    


if st.session_state["authenticated"] and page == "Survey":
    st.sidebar.success("Logged in!")
    username = st.session_state.get("username")
    st.write("Answer the questions below to receive tailored learning recommendations.")

    materials_options = ["Notes", "Hands-on Activities", "Podcasts", "Books", "Videos"]
    retention_options = ["Crammer", "Deep Learner", "Forgetting-Prone"]
    env_options = ["Low-Stress", "Structured", "Flexible", "Challenge-Oriented"]

    with st.form("survey_form"):
        visual_val = 1 if st.radio("Do you learn better visually?", ["Yes", "No"]) == "Yes" else 0
        auditory_val = 1 if st.radio("Do you prefer auditory learning?", ["Yes", "No"]) == "Yes" else 0
        kinesthetic_val = 1 if st.radio("Do you enjoy hands-on activities?", ["Yes", "No"]) == "Yes" else 0
        read_write_val = 1 if st.radio("Do you prefer learning by reading and writing?", ["Yes", "No"]) == "Yes" else 0
        preferred_material_val = encode_categorical(st.selectbox("What kind of study materials do you find most helpful?", materials_options), materials_options)

        revise_val = st.slider("How many times do you revise your material weekly?", 0, 20, 0) / 100.0
        retention_type_val = encode_categorical(st.selectbox("How would you describe your retention style?", retention_options), retention_options)
        immediate_score_val = st.number_input("What's your score for immediate retention tests?", 0, 100, 0) / 100.0
        one_week_score_val = st.number_input("What's your score for 1-week retention tests?", 0, 100, 0) / 100.0
        one_month_score_val = st.number_input("What's your score for 1-month retention tests?", 0, 100, 0) / 100.0

        math_val = st.number_input("What's your score in Math?", 0, 100, 0) / 100.0
        science_val = st.number_input("What's your score in Science?", 0, 100, 0) / 100.0
        literature_val = st.number_input("What's your score in Literature?", 0, 100, 0) / 100.0
        history_val = st.number_input("What's your score in History?", 0, 100, 0) / 100.0
        overall_perf_val = encode_categorical(st.selectbox("How would you rate your overall performance?", ["High", "Medium", "Low"]), ["High", "Medium", "Low"])

        assign1_val = st.slider("How many hours do you typically spend completing an academic task or project?", 0, 100, 0) / 1000.0
        assign2_val = st.slider("How many hours do you spend preparing for a test or exam?", 0, 100, 0) / 1000.0
        assign3_val = st.slider("How many hours do you dedicate to group projects or collaborative assignments?", 0, 100, 0) / 1000.0
        avg_assign_time_val = (assign1_val + assign2_val + assign3_val) / 3.0
        assign1_on_time_val = 1 if st.radio("Do you complete academic tasks or projects on time?", ["Yes", "No"]) == "No" else 0
        assign2_on_time_val = 1 if st.radio("Do you prepare for tests or exams well in advance?", ["Yes", "No"]) == "No" else 0
        assign3_on_time_val = 1 if st.radio("Do you meet deadlines for group projects or collaborative assignments?", ["Yes", "No"]) == "No" else 0

        logins_val = st.slider("How many times do you log in to the platform weekly?", 0, 50, 0) / 100.0
        content_reads_val = st.slider("How many pieces of content do you read weekly?", 0, 50, 0) / 100.0
        forum_reads_val = st.slider("How many forum posts do you read weekly?", 0, 50, 0) / 100.0
        forum_posts_val = st.slider("How many forum posts do you write weekly?", 0, 50, 0) / 100.0
        quiz_reviews_val = st.slider("How many times do you review quizzes before submission?", 0, 50, 0) / 100.0

        learning_env_val = encode_categorical(st.selectbox("What type of learning environment do you prefer?", env_options), env_options)
        env_perf_val = encode_categorical(st.selectbox("How would you rate your performance under this environment?", ["High", "Medium", "Low"]), ["High", "Medium", "Low"])

        submitted = st.form_submit_button("Get Recommendation")

    if submitted:
        user_features = np.array([
            0.0, visual_val, auditory_val, kinesthetic_val, read_write_val,
            preferred_material_val, 0.0, logins_val, content_reads_val,
            forum_reads_val, forum_posts_val, quiz_reviews_val, assign1_on_time_val,
            assign2_on_time_val, assign3_on_time_val, assign1_val, assign2_val,
            assign3_val, avg_assign_time_val, 0.5, 0.0, retention_type_val,
            immediate_score_val, one_week_score_val, one_month_score_val,
            revise_val, 0.0, 0.5, math_val, science_val, literature_val,
            history_val, overall_perf_val, 0.0, 0.5, 0.5, 0.5, 0.5,
            0.0, learning_env_val, 0.5, 0.5
        ]).reshape(1, -1)

        if scaler is None or scaler.n_features_in_ != user_features.shape[1]:
            st.warning("Creating new scaler for the current feature set...")
            columns = ['Dummy'] * user_features.shape[1]
            dummy_data = pd.DataFrame(np.zeros((10, len(columns))), columns=columns)
            scaler = MinMaxScaler()
            scaler.fit(dummy_data)
            joblib.dump(scaler, 'scaler.pkl')

        user_features_scaled = scaler.transform(user_features)
        latent_rep = clustering_encoder.predict(user_features_scaled)
        user_cluster = kmeans_model.predict(latent_rep)[0]

        cluster_recommendations = {
            0: "It seems you respond well to structured materials and repetition. Consider organized notes and regular reviews.",
            1: "Hands-on activities might help you engage and retain information more effectively.",
            2: "Combine visual aids with frequent short quizzes to strengthen your retention over time.",
            3: "Managing your time carefully and studying in short, focused sessions could boost your performance.",
            4: "Try studying in groups or engaging with discussion forums to enhance motivation and comprehension.",
            5: "Deep learning strategies, such as teaching someone else or summarizing your notes, may help solidify knowledge.",
            6: "Mixing various materials—reading, writing, listening—could give you a more well-rounded learning experience.",
            7: "Structured study guides and planning your assignments ahead of time can improve your submission habits.",
            8: "Reflect frequently on your learning progress and adjust your environment and strategies based on feedback.",
            9: "Don't hesitate to experiment with different study environments and materials to find what works best."
        }

        base_recommendation = cluster_recommendations.get(user_cluster, "Keep experimenting with different strategies until you find a perfect fit.")
        expanded_recommendation = generate_expanded_recommendation(base_recommendation, user_cluster)

        st.subheader(" Recommendation")
        st.write(expanded_recommendation)

        answers = {
            "visual_val": visual_val,
            "auditory_val": auditory_val,
            "kinesthetic_val": kinesthetic_val,
            "read_write_val": read_write_val,
            "preferred_material_val": preferred_material_val,
        }

        save_user_history(username, answers, user_cluster, expanded_recommendation)
        history_file = generate_history_file(username)
        if history_file:
            st.download_button(
    label="Download History",
    data=history_file,
    file_name=f"{username}_history.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

        else:
            st.error("Failed to generate history file.")
else:
    if page == "Survey":
        st.warning("Please log in to access the survey.")
# Add these imports at the top of the file, along with the existing imports
import textwrap
import PyPDF2
from docx import Document

def process_uploaded_file(uploaded_file):
    """
    Process the uploaded file and extract its content.
    """
    try:
        # Process PDF files
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text if text.strip() else None  # Return None if no content is extracted

        # Process Word documents
        elif uploaded_file.name.endswith(".docx"):
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
            return text if text.strip() else None  # Return None if no content is extracted

        else:
            return None  # Unsupported file format

    except Exception as e:
        st.sidebar.error(f"Error processing file: {str(e)}")  # Display error in sidebar
        return None



def generate_chatbot_response(conversation_history, user_input, context=None):
    """
    Generate a chatbot response using Groq API with context-aware prompting.
    """
    system_prompt = (
        "You are a helpful AI learning assistant specialized in personalized education strategies. "
        "Your goal is to provide supportive, encouraging, and practical advice to students. "
        "If the conversation relates to uploaded documents, analyze the content when explicitly asked. "
        "Otherwise, do not assume the document context unless specifically mentioned."
    )

    if context:
        system_prompt += f"\n\nDocument Context:\n{context[:1500]}"  # Include first 1500 characters for context

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([
        {"role": "user" if msg['role'] == 'user' else 'assistant', 'content': msg['content']}
        for msg in conversation_history
    ])
    messages.append({"role": "user", "content": user_input})

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1,
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Sorry, I encountered an error: {e}"


def chatbot_sidebar():
    """
    Create a chatbot sidebar in the Streamlit application with explicit user interaction for uploaded files.
    """
    # Initialize session state for chat history and file content
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'file_content' not in st.session_state:
        st.session_state.file_content = None

    # Sidebar for file upload
    st.sidebar.write("Upload a PDF or Word document for personalized learning insights:")
    uploaded_file = st.sidebar.file_uploader("Upload File", type=["pdf", "docx"])

    # Process and save the uploaded file content
    if uploaded_file is not None:
        file_content = process_uploaded_file(uploaded_file)
        if file_content:
            st.session_state.file_content = file_content
            st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")
        else:
            st.sidebar.error("Failed to process the uploaded document. Please try again.")

    # Chatbot interaction
    with st.sidebar.expander("🤖 Chatbot", expanded=True):
        # Add "Reset Chat History" button
        if st.button("Reset Chat History"):
            st.session_state.chat_history = []
            st.sidebar.success("Chat history has been reset.")

        # Display conversation history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        # User input
        user_input = st.chat_input("Ask me about the uploaded file or anything else...")
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Generate chatbot response
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    # Pass file content only if explicitly mentioned and exists
                    if uploaded_file is not None and st.session_state.file_content:
                        if "file" in user_input.lower() or "document" in user_input.lower():
                            context = st.session_state.file_content
                        else:
                            context = None
                    else:
                        context = None

                    response = generate_chatbot_response(
                        st.session_state.chat_history[:-1],  # Exclude the latest message
                        user_input,
                        context
                    )
                    st.markdown(response)

            # Add AI response to history
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})

    





# Modify the existing Streamlit GUI section
if st.session_state["authenticated"] and page == "Survey":
    # Add chatbot sidebar alongside existing survey content
    chatbot_sidebar()
    
    
    

    # RUNpython -m streamlit run 490-FINAL/app.py

import PyPDF2
from docx import Document




