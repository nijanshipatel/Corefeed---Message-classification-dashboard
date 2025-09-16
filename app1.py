import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Path to your trained model
MODEL_PATH = r"D:/projects/Message filtering/priority_message_model/content/priority_message_model"

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Custom mapping: 0 → casual, 1 → spam, 2 → important
id2label = {0: "casual", 1: "spam", 2: "important"}

# Priority order
priority_order = {"important": 2, "casual": 1, "spam": 0}

# Notification colors
priority_colors = {
    "important": "#ff4d4d",   # Red
    "casual": "#4CAF50",      # Green
    "spam": "#607D8B"         # Grey
}

st.set_page_config(page_title="Priority Notifications", layout="centered")
st.markdown("<h2 style='text-align:center;'>📱 Smart Message Notifications</h2>", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- Input box --------------------
with st.form(key="msg_form", clear_on_submit=True):
    message = st.text_input("Type a message:")
    submitted = st.form_submit_button("Send")

if submitted and message.strip():
    inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    label = id2label[pred]

    # Save message
    st.session_state.messages.append((message, label))

# -------------------- Display notifications --------------------
if st.session_state.messages:
    # Sort by priority
    sorted_msgs = sorted(
        st.session_state.messages,
        key=lambda x: priority_order[x[1]],
        reverse=True
    )

    st.subheader("🔔 Notifications")

    for msg, label in sorted_msgs:
        st.markdown(
            f"""
            <div style="
                border:1px solid #ccc;
                border-radius:10px;
                background-color:white;
                padding:12px;
                margin:10px 0;
                box-shadow:0px 4px 6px rgba(0,0,0,0.1);
                font-family:Arial;
                max-width:90%;">
                <div style="font-size:14px; color:grey;">New Message</div>
                <div style="font-size:16px; font-weight:bold; color:{priority_colors[label]};">
                    {label.upper()}
                </div>
                <div style="font-size:15px; margin-top:5px;">{msg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
