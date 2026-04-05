import streamlit as st
import json
import joblib
import os
from datetime import datetime

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Depression Detection", layout="centered")

# ----------------------------
# FILE PATHS
# ----------------------------
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

# ----------------------------
# LOAD MODEL
# ----------------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.error("Model or vectorizer file missing!")
    st.stop()

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# ----------------------------
# LOAD / SAVE USERS
# ----------------------------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

# ----------------------------
# LOAD / SAVE HISTORY
# ----------------------------
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

users = load_users()
history_data = load_history()

# ----------------------------
# SESSION STATE
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if "user" not in st.session_state:
    st.session_state.user = None

# ----------------------------
# LOGIN PAGE
# ----------------------------
if st.session_state.page == "login":
    st.markdown("## 🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.user = username
                st.session_state.page = "app"
                st.rerun()
            else:
                st.error("Invalid credentials")

    with col2:
        if st.button("Register"):
            st.session_state.page = "register"
            st.rerun()

# ----------------------------
# REGISTER PAGE
# ----------------------------
elif st.session_state.page == "register":
    st.markdown("## 📝 Create Account")

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Create Account"):
            if new_user in users:
                st.warning("User already exists")
            elif new_user.strip() == "" or new_pass.strip() == "":
                st.warning("Fill all fields")
            else:
                users[new_user] = new_pass
                save_users(users)
                st.success("Account created!")
                st.session_state.page = "login"
                st.rerun()

    with col2:
        if st.button("Back"):
            st.session_state.page = "login"
            st.rerun()

# ----------------------------
# MAIN APP
# ----------------------------
elif st.session_state.page == "app":

    st.markdown(f"## 🧠 Welcome, {st.session_state.user}")
    
    menu = st.sidebar.radio("Menu", ["Predict", "History", "Logout"])

    # ------------------------
    # PREDICT PAGE
    # ------------------------
    if menu == "Predict":
        st.subheader("Depression Detection")

        user_input = st.text_area("Enter your text:")

        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter text")
            else:
                text_vec = vectorizer.transform([user_input])
                prediction = model.predict(text_vec)[0]

                result = "Depressive" if prediction == 1 else "Non-Depressive"

                # Show result
                if prediction == 1:
                    st.error("⚠️ The text indicates depression")
                else:
                    st.success("✅ The text does not indicate depression")

                # Save history
                user = st.session_state.user
                entry = {
                    "text": user_input,
                    "result": result,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                if user not in history_data:
                    history_data[user] = []

                history_data[user].append(entry)
                save_history(history_data)

    # ------------------------
    # HISTORY PAGE
    # ------------------------
    elif menu == "History":
        st.subheader("📜 Your Prediction History")

        user = st.session_state.user

        if user not in history_data or len(history_data[user]) == 0:
            st.info("No history found")
        else:
            for item in reversed(history_data[user]):
                st.markdown("---")
                st.write(f"🕒 {item['time']}")
                st.write(f"💬 Text: {item['text']}")
                st.write(f"📊 Result: {item['result']}")

    # ------------------------
    # LOGOUT
    # ------------------------
    elif menu == "Logout":
        st.session_state.user = None
        st.session_state.page = "login"
        st.rerun()