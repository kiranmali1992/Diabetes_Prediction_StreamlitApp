
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
std = StandardScaler()

#Loading the model

loaded_model = pickle.load(open("trained_model.sav", "rb"))

def Diabetes_prediction(input_data):

    #convert it into array
    input_data = np.array(input_data).reshape(1,-1)

    #Apply Standardization
    #input_data[:,:] = std.fit_transform(input_data[:,:])

    #Prediction of output
    Prediction = loaded_model.predict(input_data)

    if(Prediction[0] == 1):
        
      return "Person is having Diabetes"
    
    else:
        
      return "Person is not having Diabetes"

def main():

    # Giving title
    st.title("Diabetes Prediction")

    #Getting input from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("BloodPressure value")
    SkinThickness = st.text_input("SkinThickness value")
    Insulin = st.text_input("Insulin value")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
    Age = st.text_input("Age")

    #Code for prediction
    diagnosis = ""

    #Creating Button
    if st.button("Diabetes test result"):
        diagnosis = Diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)

if __name__ == "__main__":
    main()


    
