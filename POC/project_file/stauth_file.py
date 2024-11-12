import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load credentials from the YAML file
with open(r"E:\poc_files\POC\project_file\credentials.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize the authenticator with all required arguments
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized'],
    "asus401kok"  # Add a unique key here for your app, e.g., "my_secret_key"
)

name, authentication_status, username = authenticator.login('Login', 'main')
if authentication_status:
    authenticator.logout('Logout', 'main')
    if username == 'adam':
        st.write(f'Welcome *{name}*')
        st.title('Application 1')
    elif username == 'chris':
        st.write(f'Welcome *{name}*')
        st.title('Application 2')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')