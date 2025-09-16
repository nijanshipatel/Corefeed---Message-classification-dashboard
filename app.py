# import streamlit as st
# import torch
# from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
# MODEL_PATH = r"D:/projects/Message filtering/priority_message_model/content/priority_message_model"

# tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)


# st.title("📩 Priority Message Dashboard")

# # Input box for single message
# message = st.text_area("Enter a message to classify:")

# if st.button("Classify"):
#     inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         pred = torch.argmax(outputs.logits, dim=1).item()

#     labels = ["Low/Spam", "Normal", "High Priority"]
#     st.success(f"Prediction: **{labels[pred]}**")

# # Batch ranking section
# st.header("📊 Rank Multiple Messages")
# uploaded = st.file_uploader("Upload a text file with one message per line", type=["txt"])

# if uploaded:
#     lines = uploaded.read().decode("utf-8").splitlines()
#     results = []

#     for line in lines:
#         if line.strip():  # skip empty lines
#             inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 pred = torch.argmax(outputs.logits, dim=1).item()
#             results.append((line, pred))

#     # Sort by priority (High > Normal > Low/Spam)
#     results.sort(key=lambda x: x[1], reverse=True)

#     labels = ["Low/Spam", "Normal", "High Priority"]

#     st.subheader("🔝 Ranked Messages")
#     for rank, (msg, label) in enumerate(results, 1):
#         st.write(f"**{rank}. {msg}** → {labels[label]}")


# import streamlit as st
# import torch
# from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# MODEL_PATH = r"D:/projects/Message filtering/priority_message_model/content/priority_message_model"

# tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# # Use config mapping if exists, else custom labels
# if hasattr(model.config, "id2label") and model.config.id2label:
#     id2label = model.config.id2label
# else:
#     id2label = {0: "Low/Spam", 1: "Normal", 2: "High Priority"}

# st.title("📩 Priority Message Dashboard")

# # Input box for single message
# message = st.text_area("Enter a message to classify:")

# if st.button("Classify"):
#     inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)

#     with torch.no_grad():
#         outputs = model(**inputs)
#         pred = torch.argmax(outputs.logits, dim=1).item()

#     st.success(f"Prediction: **{id2label[pred]}**")

# # Batch ranking section
# st.header("📊 Rank Multiple Messages")
# uploaded = st.file_uploader("Upload a text file with one message per line", type=["txt"])

# if uploaded:
#     lines = uploaded.read().decode("utf-8").splitlines()
#     results = []

#     for line in lines:
#         if line.strip():  # skip empty lines
#             inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 pred = torch.argmax(outputs.logits, dim=1).item()
#             results.append((line, pred))

#     # Sort by priority (High > Normal > Low/Spam)
#     results.sort(key=lambda x: x[1], reverse=True)

#     st.subheader("🔝 Ranked Messages")
#     for rank, (msg, label) in enumerate(results, 1):
#         st.write(f"**{rank}. {msg}** → {id2label[label]}")


# import streamlit as st
# import torch
# from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# # Path to your trained model
# MODEL_PATH = r"D:/projects/Message filtering/priority_message_model/content/priority_message_model"

# # Load tokenizer and model
# tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
# model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# # Custom mapping: 0 → casual, 1 → spam, 2 → important
# id2label = {
#     0: "casual",
#     1: "spam",
#     2: "important"
# }

# # Priority order for sorting: important > casual > spam
# priority_order = {"important": 2, "casual": 1, "spam": 0}

# st.title("📩 Priority Message Dashboard")

# # -------------------- Single message classification --------------------
# message = st.text_area("Enter a message to classify:")

# if st.button("Classify"):
#     if message.strip():
#         inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#             pred = torch.argmax(outputs.logits, dim=1).item()
#         label = id2label[pred]
#         st.success(f"Prediction: **{label}**")
#     else:
#         st.warning("Please enter a message to classify.")

# # -------------------- Batch ranking --------------------
# st.header("📊 Rank Multiple Messages")
# uploaded = st.file_uploader("Upload a text file with one message per line", type=["txt"])

# if uploaded:
#     lines = uploaded.read().decode("utf-8").splitlines()
#     results = []

#     for line in lines:
#         if line.strip():
#             inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
#             with torch.no_grad():
#                 outputs = model(**inputs)
#                 pred = torch.argmax(outputs.logits, dim=1).item()
#             label = id2label[pred]
#             results.append((line, label))

#     # Sort messages by priority: important > casual > spam
#     results.sort(key=lambda x: priority_order[x[1]], reverse=True)

#     st.subheader("🔝 Ranked Messages")
#     st.table([
#         {"Rank": i+1, "Message": msg, "Prediction": label} 
#         for i, (msg, label) in enumerate(results)
#     ])

import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Path to your trained model
MODEL_PATH = r"D:/projects/Message filtering/priority_message_model/content/priority_message_model"

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

# Custom mapping: 0 → casual, 1 → spam, 2 → important
id2label = {
    0: "casual",
    1: "spam",
    2: "important"
}

# Priority order for sorting: important > casual > spam
priority_order = {"important": 2, "casual": 1, "spam": 0}

# Colors for chat bubbles
priority_colors = {
    "important": "#ff6961",  # Red for important
    "casual": "#77dd77",     # Green for casual
    "spam": "#aec6cf"        # Grey-blue for spam
}

st.set_page_config(page_title="Priority Chat Dashboard", layout="centered")
st.markdown("<h2 style='text-align:center;'>📱 Priority Message Dashboard</h2>", unsafe_allow_html=True)

# -------------------- Single message classification --------------------
message = st.text_area("Enter a message to classify:")

if st.button("Classify"):
    if message.strip():
        inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        label = id2label[pred]

        # Show as chat bubble
        st.markdown(
            f"""
            <div style="
                background-color:{priority_colors[label]};
                border-radius:12px;
                padding:10px 15px;
                margin:8px 0;
                max-width:70%;
                color:white;
                font-size:16px;">
                <b>{label.upper()}</b>: {message}
            </div>
            """, unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a message to classify.")

# -------------------- Batch ranking --------------------
st.header("📊 Rank Multiple Messages")
uploaded = st.file_uploader("Upload a text file with one message per line", type=["txt"])

if uploaded:
    lines = uploaded.read().decode("utf-8").splitlines()
    results = []

    for line in lines:
        if line.strip():
            inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            label = id2label[pred]
            results.append((line, label))

    # Sort messages by priority
    results.sort(key=lambda x: priority_order[x[1]], reverse=True)

    st.subheader("🔝 Ranked Messages")

    # Show ranked messages as chat bubbles
    for rank, (msg, label) in enumerate(results, 1):
        st.markdown(
            f"""
            <div style="
                background-color:{priority_colors[label]};
                border-radius:12px;
                padding:10px 15px;
                margin:8px 0;
                max-width:70%;
                color:white;
                font-size:16px;">
                <b>{rank}. {label.upper()}</b>: {msg}
            </div>
            """, unsafe_allow_html=True
        )

