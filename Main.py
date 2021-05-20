##Decription : detect the someone has diebetes or not using web and machine learning
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

##Create the title and the subtitle

st.write("""
# Diabetes Detection By PREM VARMA
Detect if Someone has diabetes using machine learning and python !
""")

#open and disply

image = Image.open('F:/Machine Learning/Diabetes Detection/Diabetes.jpg')
st.image(image,caption='ML',use_column_width=True)

#get the data
df =pd.read_csv('F:/Machine Learning/Diabetes Detection/diabetes.csv')

#set a subheader on webapp
st.subheader('Data Information:')
#show data as a table
st.dataframe(df)
#show staticstics
st.write(df.describe())

#show data
chart = st.bar_chart(df)
#split data into x and y
x = df.iloc[:,0:8].values
y = df.iloc[:,-1].values
#split into 75% training and 25% test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#get the fetecher ip from user
def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    Glucose = st.sidebar.slider('Glucose',0,199,117)
    BloodPressure = st.sidebar.slider('BloodPressure',0,122,72)
    SkinThickness = st.sidebar.slider('SkinThickness',0,99,23)
    Insulin = st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    Age = st.sidebar.slider('Age',21,81,29)

    # store a dict in variable
    user_data ={' Pregnancies':Pregnancies,
                'Glucose': Glucose ,
                'BloodPressure':BloodPressure,
                'SkinThickness': SkinThickness,
                'Insulin': Insulin,
                'BMI': BMI,
                'DPF':DPF,
                'Age':Age
                }
    #transform data in dataframe
    features = pd.DataFrame(user_data,index=[0])
    return features

#store input in variable
user_input = get_user_input()
#set a subheader disply user ip

st.subheader('User Input')
st.write(user_input)

#create and train model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#show the model
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test))*100)+'%')

#store the model prediction

prediction = RandomForestClassifier.predict(user_input)

#set a subheader and disply classification
st.subheader('Classification:')
st.write(prediction)
