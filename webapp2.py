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

st.markdown("""
<meta name="description" content="Machine Learning based Literacy Rate Forecasting System for Indian States">
<meta name="keywords" content="Machine Learning, Literacy Rate, India, Forecasting, Streamlit">
<meta name="author" content="Simran Shaikh">
""", unsafe_allow_html=True)

# ---------------- BACKGROUND ----------------

st.markdown(
"""
<style>
.stApp {
    background-image: url("https://wallpaperboat.com/wp-content/uploads/2019/10/free-website-background-01.jpg");
    background-size: cover;
}
</style>
""",
unsafe_allow_html=True
)

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
    data = c.fetchall()
    return data

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

create_usertable()

# ---------------- SESSION STATE ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- HEADER ----------------

st.title("📊 Indian Literacy Rate Forecasting System")
st.subheader("Machine Learning Based Prediction of Literacy Rates in Indian States")

# ---------------- MENU ----------------

menu = ["Home","Login","Signup","About"]
choice = st.sidebar.selectbox("Menu",menu)

# ---------------- HOME ----------------

if choice == "Home":

    st.write("""
    ### Welcome

    This system forecasts literacy rates for Indian states using Machine Learning models.

    **Models Implemented**
    - Linear Regression
    - Random Forest
    - Decision Tree
    - Gradient Boosting

    Upload your dataset and evaluate model performance interactively.
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
        st.info("Go to Login Menu")

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

    # -------- AFTER LOGIN --------

    if st.session_state.logged_in:

        st.sidebar.success("Logged In")

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        page = st.sidebar.radio("Navigation",["Dashboard","Prediction"])

        # -------- DASHBOARD --------

        if page == "Dashboard":

            st.header("Dataset Dashboard")

            data_file = st.file_uploader("Upload Dataset",type=["csv"])

            if data_file is not None:

                data = pd.read_csv(data_file)

                st.subheader("Dataset Preview")
                st.dataframe(data)

                st.subheader("Dataset Statistics")
                st.write(data.describe())

                st.subheader("Numeric Feature Visualization")
                st.bar_chart(data.select_dtypes(include=np.number))

        # -------- PREDICTION --------

        if page == "Prediction":

            st.header("Literacy Rate Prediction")

            data_file = st.file_uploader("Upload CSV file",type=["csv"])

            if data_file is not None:

                data = pd.read_csv(data_file)

                st.subheader("Dataset Preview")
                st.write(data)

                data.replace("na",np.nan,inplace=True)
                data.dropna(inplace=True)

                st.subheader("Processed Data")
                st.write(data)

                column_options = list(data.columns[2:])
                columns = st.multiselect("Select Features",options=column_options,default=column_options[:6])

                target = st.selectbox("Select Target Variable",data.columns)

                X = data[columns]
                y = data[target]

                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

                models = {
                    "Linear Regression":LinearRegression(),
                    "Random Forest":RandomForestRegressor(),
                    "Decision Tree":DecisionTreeRegressor(),
                    "Gradient Boosting":GradientBoostingRegressor()
                }

                model_choice = st.selectbox("Select Model",list(models.keys()))

                model = models[model_choice]

                model.fit(X_train,y_train)

                y_pred = model.predict(X_test)

                st.subheader("Model Performance")

                st.write("R2 Score:",r2_score(y_test,y_pred))
                st.write("MSE:",mean_squared_error(y_test,y_pred))
                st.write("MAE:",mean_absolute_error(y_test,y_pred))

                fig,ax = plt.subplots()

                ax.plot(y_test.values,label="Actual")
                ax.plot(y_pred,label="Predicted")
                ax.legend()

                st.pyplot(fig)

                # -------- MODEL COMPARISON --------

                st.subheader("Model Accuracy Comparison")

                scores = {}

                for name,m in models.items():

                    m.fit(X_train,y_train)

                    pred = m.predict(X_test)

                    scores[name] = r2_score(y_test,pred)

                st.bar_chart(scores)

                # -------- USER INPUT PREDICTION --------

                st.subheader("Predict Next Year Literacy Rate")

                user_input = st.text_input("Enter feature values separated by commas")

                if st.button("Predict"):

                    values = list(map(float,user_input.split(",")))

                    test_df = pd.DataFrame([values],columns=columns)

                    prediction = model.predict(test_df)

                    st.success(f"Predicted Literacy Rate: {prediction[0]:.2f}%")

                # -------- DOWNLOAD REPORT --------

                result_df = pd.DataFrame({
                    "Actual":y_test,
                    "Predicted":y_pred
                })

                csv = result_df.to_csv(index=False)

                st.download_button(
                    label="Download Prediction Report",
                    data=csv,
                    file_name="prediction_report.csv",
                    mime="text/csv"
                )

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

