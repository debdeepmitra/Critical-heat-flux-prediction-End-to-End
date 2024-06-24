import streamlit as st
import pandas as pd
import numpy as np
import joblib
from helper import author_encoder, geometry_encoder, get_prediction

model = joblib.load(r'Model\model.joblib')

st.set_page_config(page_title ='Critical Heat Flux Prediction', page_icon = '🔥', layout ='wide')
st.markdown("<h1 style = 'text-align: center;'>Critical Heat Flux Prediction 🔥</h1>", unsafe_allow_html=True)

options_author = ['Thompson', 'Janssen', 'Weatherhead', 'Beus', 'Williams', 'Richenderfer', 'Mortimore', 'Peskov', 'Kossolapov', 'Inasaka']
options_geometry = ['tube', 'annulus', 'plate']

def main():
    with st.form('prediction_form'):

        st.subheader('Enter the input to following features:')
 
        author = st.selectbox("Choose the author", options=options_author)
        geometry = st.selectbox("Choose the geometry", options=options_geometry)
        pressure = st.number_input('Pressure: (MPa)', 0.10, 20.68, value='min', step=.01, format='%.2f')
        mass_flux = st.number_input('Mass_Flux: (kg/m2-s)', 0.0, 7975.0, value='min', step=.01, format='%.2f')
        x_e_out = st.number_input('x_e_out', -0.8667, 0.2320, value='min', step=.01, format='%.2f')
        D_e = st.number_input('D_e: (mm)', 1.0, 37.5, value='min', step=.01, format='%.2f')
        D_h = st.number_input('D_h: (mm)', 1.0, 120.0, value='min', step=.01, format='%.2f')
        length = st.number_input('Length: (mm)', 10.0, 3048.0, value='min', step=.01, format='%.2f')
 
        submit = st.form_submit_button('Predict Yield')

    if submit:

        encoded_author=author_encoder(author)
        encoded_geometry=geometry_encoder(geometry)

        data = np.array([encoded_author, encoded_geometry, pressure, mass_flux, x_e_out, D_h, D_e, length]).reshape(1,-1)
        #data = [clonesize, bumbles, andrena, osmia, AverageOfUpperTRange, AverageOfLowerTRange, AverageRainingDays]
        pred = get_prediction(data, model)

        st.write(f'The predicted heat flux is: {pred:.2f}')


if __name__ == "__main__":
    main()