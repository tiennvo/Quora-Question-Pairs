import streamlit as st
import helper
import pickle
import Chatopenai

model = pickle.load(open('D:/Python/Quora/Quora/quora-question-pairs/model.pkl', 'rb'))

st.header('Qoura Question Pairs')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')


