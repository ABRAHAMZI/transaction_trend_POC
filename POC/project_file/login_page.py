import streamlit as st
import bcrypt

# Hash the password once and store it securely (only done once)
stored_hash = bcrypt.hashpw("walk4LIFE".encode('utf-8'), bcrypt.gensalt())

# Function to handle authentication with password hashing
def authenticate():
    # Create the login form
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")

    # When the user clicks the "Login" button
    if st.button("Login"):
        # Check if the entered username and password match
        if username == "adam" and bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid username or password.")
            return False

# Main code to run the Streamlit app
def main():
    # Show the login form and authenticate
    if authenticate():
        # If authentication is successful, show the rest of the app
        st.title("Welcome to the Streamlit App!")
        st.write("You are now logged in.")
        
        # Place the rest of your app's code here
        st.write(f"This is the trend for {username}.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
