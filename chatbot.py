import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file with error handling
file_path = r"C:\Users\HITISH\Downloads\Chatbot_using_NLP_AICTE_Cycle4-main\Chatbot_using_NLP_AICTE_Cycle4-main\ImplementationofChatBot.ipynb"
try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error(f"Error loading intents.json: {e}")
    st.stop()

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't understand."

# Initialize chat history in Streamlit session state
if "history" not in st.session_state:
    st.session_state["history"] = []

def main():
    st.title("NLP-Based Chatbot")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! Type a message and press Enter to chat.")

        # Ensure chat log CSV file exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        user_input = st.text_input("You:", key="user_input")

        if user_input.strip():  # Prevent empty inputs
            response = chatbot(user_input)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Store in session history
            st.session_state["history"].append(("You:", user_input))
            st.session_state["history"].append(("Chatbot:", response))

            # Display chat history
            for msg in st.session_state["history"]:
                st.text(msg)

            # Save chat to CSV
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thanks for chatting! Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader, None)  # Skip header
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.subheader("About the Chatbot")
        st.write("""
        This chatbot uses **Natural Language Processing (NLP)** techniques to understand and respond to user queries.
        It is built using:
        - **TF-IDF Vectorization** for text processing
        - **Logistic Regression** for intent classification
        - **Streamlit** for the chatbot interface
        """)

if __name__ == '__main__':
    main()
