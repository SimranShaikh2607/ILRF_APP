#python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Indian Literacy Rate Forecasting",
    page_icon="📊",
    layout="wide"
)

# ---------------- CLEAN PROFESSIONAL DARK UI ----------------

st.markdown("""
<style>

/* ---------- BASE ---------- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #111827;
    color: #e5e7eb;
    font-family: 'Segoe UI', sans-serif;
}

/* ---------- LAYOUT ---------- */
.main, .block-container {
    background: transparent !important;
    padding: 2rem 3rem;
}

/* ---------- TITLE ---------- */
h1 {
    text-align: left;
    font-size: 28px;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 10px;
}

/* ---------- HEADINGS ---------- */
h2, h3 {
    color: #d1d5db;
    font-weight: 500;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #1f2937;
}

/* ---------- SIMPLE SECTIONS ---------- */
div[data-testid="stVerticalBlock"] > div {
    padding: 12px 0;
    border-bottom: 1px solid #1f2937;
}

/* ---------- BUTTON ---------- */
.stButton>button {
    background: #374151;
    color: #e5e7eb;
    border-radius: 6px;
    padding: 6px 12px;
    border: 1px solid #4b5563;
    font-size: 14px;
}

.stButton>button:hover {
    background: #4b5563;
}

/* ---------- INPUT ---------- */
.stTextInput>div>div>input {
    background-color: #020617;
    color: #e5e7eb;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 6px;
}

/* ---------- SELECT ---------- */
.stSelectbox>div>div {
    background-color: #020617;
    color: #e5e7eb;
    border: 1px solid #374151;
    border-radius: 6px;
}

/* ---------- FILE UPLOADER ---------- */
.stFileUploader {
    border: 1px dashed #374151;
    padding: 8px;
    border-radius: 6px;
}

/* ---------- SUCCESS / ERROR ---------- */
.stSuccess {
    color: #22c55e;
}

.stError {
    color: #ef4444;
}

/* ---------- REMOVE HEADER / FOOTER ---------- */
header, footer {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------

conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT,email TEXT,password TEXT)')

def add_userdata(username,email,password):
    c.execute('INSERT INTO users(username,email,password) VALUES (?,?,?)',(username,email,password))
    conn.commit()

def login_user(email,password):
    c.execute('SELECT * FROM users WHERE email=? AND password=?',(email,password))
    return c.fetchall()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

create_usertable()

# ---------------- SESSION STATE ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- HEADER ----------------

st.title("📊 Indian Literacy Rate Forecasting System")

# ---------------- MENU ----------------

menu = ["Home","Login","Signup","About"]
choice = st.sidebar.selectbox("Menu",menu)

# ---------------- HOME ----------------

if choice == "Home":
    st.write("""
    ### Welcome
    Forecast literacy rates using Machine Learning models.

    **Models Implemented**
    - Linear Regression
    - Random Forest
    - Decision Tree
    - Gradient Boosting
    """)

# ---------------- SIGNUP ----------------

elif choice == "Signup":
    st.subheader("Create Account")

    new_user = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password",type='password')

    if st.button("Signup"):
        add_userdata(new_user,new_email,hash_password(new_password))
        st.success("Account Created Successfully")

# ---------------- LOGIN ----------------

elif choice == "Login":

    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password",type='password')

    if st.button("Login"):
        result = login_user(email,hash_password(password))

        if result:
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid Email or Password")

    if not st.session_state.logged_in:
        st.warning("Please login first to access prediction")
        st.stop()

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ---------------- PREDICTION ----------------

    st.header("Literacy Rate Prediction")

    data_file = st.file_uploader("Upload CSV file", type=["csv"])

    if data_file is not None:

        data = pd.read_csv(data_file)

        st.subheader("Dataset Preview")
        st.write(data)

        data.replace("na", np.nan, inplace=True)
        data.dropna(inplace=True)

        st.subheader("Processed Data")
        st.write(data)

        # ------ Convert categorical columns FIRST -------
data = pd.get_dummies(data, drop_first=True)

# st.subheader("Processed Data (After Encoding)")
# st.write(data)

# ------ Now select features and target ----------
columns = st.multiselect("Select Features", data.columns)
target = st.selectbox("Select Target Variable", data.columns)

# ------ Safety check ----------
if target not in data.columns:
    st.error(f"Target column '{target}' not found after processing")
    st.stop()

# ------ Use selected features ----------
if columns:
    X = data[columns]
else:
    X = data.drop(target, axis=1)

y = data[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

st.subheader("Model Performance")
st.write("R2 Score:", r2_score(y_test,y_pred))
st.write("MSE:", mean_squared_error(y_test,y_pred))
st.write("MAE:", mean_absolute_error(y_test,y_pred))

fig,ax = plt.subplots()
ax.plot(y_test.values,label="Actual")
ax.plot(y_pred,label="Predicted")
ax.legend()
st.pyplot(fig)

        # -------- USER INPUT --------

st.subheader("Predict Next Year Literacy Rate")

user_input = st.text_input("Enter feature values separated by commas")

if st.button("Predict"):
            values = list(map(float,user_input.split(",")))
            test_df = pd.DataFrame([values],columns=columns)
            prediction = model.predict(test_df)
            st.success(f"Predicted Literacy Rate: {prediction[0]:.2f}%")

# ---------------- ABOUT ----------------

elif choice == "About":

    st.header("About This Project")

    st.write("""
    **Indian Literacy Rate Forecasting System**

    This project uses Machine Learning algorithms to predict literacy rates
    of Indian states using historical data.

    **Technologies Used**
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - Matplotlib

    Developed by **Simran Shaikh**
    """)

#Add github icon

st.markdown(
"""
<div style="text-align:center; font-size:16px">

<b>Created by Simran Shaikh</b><br><br>

<a href="https://github.com/SimranShaikh2607" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="25">
</a>

</div>
""",
unsafe_allow_html=True
)

