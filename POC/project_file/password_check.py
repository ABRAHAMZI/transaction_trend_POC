import streamlit_authenticator as stauth
import warnings
import streamlit as st

st.title("MY streamlit password encrypted")

# Ignore all warnings
warnings.filterwarnings('ignore')
# List of plain-text passwords
passwords = ["asdf23", "jkl90"]

# Hash the passwords
hashed_passwords = stauth.Hasher(passwords).generate()
st.write(hashed_passwords)
print(hashed_passwords)
