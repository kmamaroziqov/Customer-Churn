import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO
import streamlit.components.v1 as components

st.title('Customer Churn Prediction')
st.write('This app predicts if the customer churns')

# Model and scaler URLs
model_url = 'https://github.com/kmamaroziqov/Customer-Churn/raw/main/churn_model.pkl'
scaler_url = 'https://github.com/kmamaroziqov/Customer-Churn/raw/main/Scaler.pkl'

# Function to load model and scaler from URLs
@st.cache_resource()
def load_model_and_scaler(model_url, scaler_url):
    # Load model
    model_response = requests.get(model_url)
    model = joblib.load(BytesIO(model_response.content))

    # Load scaler
    scaler_response = requests.get(scaler_url)
    scaler = joblib.load(BytesIO(scaler_response.content))

    return model, scaler

model, scaler = load_model_and_scaler(model_url, scaler_url)

# User inputs
Tenure = st.number_input('Enter the tenure:', min_value=0)
Complain = st.selectbox('Any complaints?', ('Yes', 'No'))
DaySinceLastOrder = st.number_input('Days since last order:', min_value=0)
CashbackAmount = st.number_input('Enter cashback amount:', min_value=0)
MaritalStatus = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
Gender = st.selectbox('Gender', ('Male', 'Female'))

st.sidebar.title('Manual')
st.sidebar.write('**Customer Tenure** - *the average time measured in years since customers initiated their contracts or business*')
st.sidebar.write('**Any Complain** - *record of whether any complain is received from customer*')
st.sidebar.write('**Days Since Last Order** - *the number of days since last order*')
st.sidebar.write('**Cashback Amount** - *the amound of cashback or discount amount received by customer*')

if st.button('Predict Churn'):
    try:
        # Prepare the input data
        input_data = pd.DataFrame({
            'Tenure': [Tenure],
            'Complain': [1 if Complain == 'Yes' else 0],
            'DaySinceLastOrder': [DaySinceLastOrder],
            'CashbackAmount': [CashbackAmount],
            'MaritalStatus': [1 if MaritalStatus == 'Single' else 2 if MaritalStatus == 'Married' else 3],
            'Gender': [1 if Gender == 'Male' else 0]
        })
        
        # Scale the input data using the pre-fitted scaler
        prepared_data = scaler.transform(input_data)
        
        # Make predictions
        prediction = model.predict(prepared_data)
        
        # Display the prediction
        text='Churn' if prediction[0] == 1 else 'No Churn'
        st.title(f'Prediction: {text}')
    except Exception as e:
        st.error(f"Error in processing input: {e}")
        
    
if 'show_iframe' not in st.session_state:
    st.session_state.show_iframe = False  # Initially, do not show the iframe

# Button to toggle the iframe display
if st.button('Show iframe'):
    st.session_state.show_iframe = True  # Show the iframe when the button is clicked

if st.session_state.show_iframe:
    # Embed the iframe using components.html for custom HTML content
    st.markdown("### Embedded Content")
    components.html("""
    <iframe src='https://nbviewer.org/github/kmamaroziqov/Customer-Churn/blob/main/Customer%20Churn.ipynb'
            width="700" 
            height="500" 
            style="overflow: auto; border: none;">
    </iframe>
    """, height=505)
    
    # Button to 'close' (hide) the iframe
    if st.button('Close iframe'):
        st.session_state.show_iframe = False  # Hide the iframe    