import streamlit as st
import pandas as pd
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import math

# Function to display the NETWORK CONGESTION dashboard
def display_network_congestion():
    network_traffic_df = pd.read_csv('C:\\Users\\kille\\OneDrive\\Documents\\Placement\\SocGen\\network_traffic.csv')
    
    # Use a wide layout for the dashboard
    st.markdown("<div style='width: 80%;'><h3>Network Congestion Dashboard</h3></div>", unsafe_allow_html=True)
    pyg_app = StreamlitRenderer(network_traffic_df, width=1400, height=400)  # Adjust width and height as needed
    pyg_app.explorer()

# Function to display the AUTOMATIC INCIDENT TRIAD dashboard
def display_automatic_incident_triad():
    display_automatic_incident_triad_df = pd.read_csv('C:\\Users\\kille\\OneDrive\\Documents\\Placement\\SocGen\\asset.csv')
    
    # Use a wide layout for the dashboard
    st.markdown("<div style='width: 80%;'><h3>AUTOMATIC INCIDENT TRIAD Dashboard</h3></div>", unsafe_allow_html=True)
    pyg_app = StreamlitRenderer(display_automatic_incident_triad_df, width=1400, height=400)  # Adjust width and height as needed
    pyg_app.explorer()
    

# Issue to resolution mapping
issue_to_resolution = {
    "System slow": "self",
    "Application crash": "self",
    "Network disconnection": "self",
    "Blue screen error": "IT support",
    "Disk error": "IT support",
    "Battery issue": "self",
    "Overheating": "IT support",
    "Software compatibility issue": "self"
}

# Initialize session state
if 'step' not in st.session_state:
    st.session_state['step'] = 0
if 'description' not in st.session_state:
    st.session_state['description'] = ''
if 'system_id' not in st.session_state:
    st.session_state['system_id'] = ''
if 'resolution_type' not in st.session_state:
    st.session_state['resolution_type'] = None

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(issue_to_resolution))

# Function to classify incidents
def classify_incident(description):
    inputs = tokenizer(description, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return predicted_class

# Sidebar buttons
if st.sidebar.button("NETWORK CONGESTION"):
    st.sidebar.write("NETWORK CONGESTION button clicked")
    # Call display_network_congestion function here

if st.sidebar.button("AUTOMATED INCIDENT TRIAGE"):
    st.session_state['step'] = 1  # Move to the first input step

# Step 1: Description input
if st.session_state['step'] == 1:
    st.subheader("Incident Triage Inputs")
    st.session_state['description'] = st.text_area("Description:", value=st.session_state['description'])
    
    if st.button("Next"):
        st.session_state['step'] = 2  # Move to the next input step

# Step 2: System ID input
if st.session_state['step'] == 2:
    system_id = st.text_input("System ID:", value=st.session_state['system_id'])
    st.session_state['system_id'] = system_id

    if st.button("Classify Incident"):
        predicted_issue = classify_incident(st.session_state['description'])
        predicted_issue_name = list(issue_to_resolution.keys())[predicted_issue]
        resolution_type = issue_to_resolution[predicted_issue_name]
        st.session_state['resolution_type'] = resolution_type
        
        st.write(f"Issue: {predicted_issue_name}, Resolution Type: {resolution_type}")
        
        if resolution_type == "IT support":
            st.session_state['step'] = 3  # Move to metrics input

# Step 3: Additional system metrics input
if st.session_state['step'] == 3:
    st.write("Additional System Metrics:")
    cpu_usage = st.number_input("CPU Usage (%):")
    memory_usage = st.number_input("Memory Usage (%):")
    disk_usage = st.number_input("Disk Usage (%):")
    network_activity = st.number_input("Network Activity (MB):")
    software_installed = st.text_input("Software Installed:")
    age = st.number_input("Age (years):")

    if st.button("Predict Time to Failure"):
        new_test_df = pd.DataFrame({
            'CPU Usage (%)': [cpu_usage],
            'Memory Usage (%)': [memory_usage],
            'Disk Usage (%)': [disk_usage],
            'Network Activity (MB)': [network_activity],
            'Software Installed': [software_installed],
            'Age (years)': [age]
        })

        # Load the model
        loaded_model = joblib.load("C:\\Users\\kille\\OneDrive\\Documents\\Placement\\SocGen\\xgboost_model.pkl")
        y_new_pred = loaded_model.predict(new_test_df)
        time_to_failure = math.floor(y_new_pred[0] / 24)  # Convert to days
        st.write(f"Predicted Time to Failure: {time_to_failure} days")

        threshold = 150  # Example threshold in hours
        if time_to_failure < threshold:
            st.write(f"ALERT: Asset ID {st.session_state['system_id']} is predicted to fail in {time_to_failure} days. Sending alert to IT support.")
    else:
        st.write("Providing self-guide...")

st.sidebar.subheader("INTERACTIVE DASHBOARD")
dashboard_option = st.sidebar.selectbox(
    "Choose an option",
    ("", "NETWORK CONGESTION", "AUTOMATED INCIDENT TRIAGE")
)

# Display the selected option in the main window
if dashboard_option == "NETWORK CONGESTION":
    display_network_congestion()
elif dashboard_option == "AUTOMATED INCIDENT TRIAGE":
    display_automatic_incident_triad()

# Floating messages
floating_message = """
<style>
    .floating-message-container {
        position: fixed;
        bottom: 10px;
        right: 0;
        width: 100%;
        overflow: hidden;
        white-space: nowrap;
    }
    .floating-message {
        display: inline-block;
        animation: float 20s linear infinite;
        white-space: nowrap;
    }
    .message-1 {
        color: lightgreen;
        padding-right: 50px;
    }
    .message-2 {
        color: yellow;
        padding-right: 50px;
    }
    @keyframes float {
        0% {
            transform: translateX(100%);
        }
        100% {
            transform: translateX(-100%);
        }
    }
    .star {
        color: red;
        font-size: 20px;
        animation: flash 1s infinite;
    }
    @keyframes flash {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0;
        }
    }
</style>

<div class="floating-message-container">
    <div class="floating-message">
        <span class="star">★</span>
        <span class="message-1">A 20.0% growth in trade volume is expected in the ASIA region on Wednesday and hence, 30 units of idle resources is requested from EU region.</span>
        <span class="star">★</span>
        <span class="message-2">A 25.0% growth in trade volume is expected in the ASIA region on Tuesday and hence, 40 units of idle resources is requested from EU region.</span>
    </div>
</div>
"""

st.markdown(floating_message, unsafe_allow_html=True)
