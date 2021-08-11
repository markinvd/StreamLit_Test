#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df =  pd.read_csv('https://raw.githubusercontent.com/markinvd/StreamLit_Test/main/water_potability.csv')
df.sample(5)


# In[3]:


df.fillna(df.mean(), inplace=True)


# In[5]:


#separando em treino e teste
X = df.drop('Potability',axis='columns')
y = df['Potability']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42, stratify = y)


# In[50]:


import sklearn


# In[45]:


from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score

model = RandomForestClassifier(n_estimators = 126,
 min_samples_split =  5,
 min_samples_leaf= 2,
 max_features= 'sqrt',
 max_depth= 44,
 bootstrap= True)
model.fit(X_train, y_train)

y_pred_rf =model.predict(X_test)


# In[44]:


import streamlit as st


# In[46]:


# Defining Prediction Function

def predict_rating(model, df):
    
    predictions_data = model.predict(df)
    
    
    return predictions_data


# In[ ]:





# In[8]:


# Writing App Title and Description

st.title('Water Quality Classifier Web App')
st.write('This is a web app to predict the water quality based on        several features that you can see in the sidebar. Please adjust the        value of each feature. After that, click on the Predict button at the bottom to        see the prediction of the classifier.')


# In[38]:


# Making Sliders and Feature Variables


ph        = st.sidebar.slider(label = 'ph', min_value = 0.0,
                        max_value = 13.9999999 ,
                        value = 7.0,
                        step = 0.05)
                        
Hardness = st.sidebar.slider(label = 'Hardness', min_value = 47.432,
                        max_value = 323.124 ,
                        value = 100.0,
                        step = 0.5)

Solids = st.sidebar.slider(label = 'Solids', min_value = 321.0,
                        max_value = 61227.0 ,
                        value = 10000.0,
                        step = 1.0)

Chloramines = st.sidebar.slider(label = 'Chloramines', min_value = 0.3520,
                        max_value = 13.127 ,
                        value = 5.0,
                        step = 0.1)

Sulfate = st.sidebar.slider(label = 'Sulfate', min_value = 129.0,
                        max_value = 481.0 ,
                        value = 250.0,
                        step = 1.0)

Conductivity = st.sidebar.slider(label = 'Conductivity', min_value = 181.0,
                        max_value = 753.0 ,
                        value = 300.0,
                        step = 1.0)

Organic_carbon = st.sidebar.slider(label = 'Organic_carbon', min_value = 2.2,
                        max_value = 28.3 ,
                        value = 10.0,
                        step = 0.1)

Trihalomethanes = st.sidebar.slider(label = 'Trihalomethanes', min_value = 0.73,
                        max_value = 124.0 ,
                        value = 40.0,
                        step = 0.1)

Turbidity = st.sidebar.slider(label = 'Turbidity', min_value = 1.45,
                        max_value = 6.73 ,
                        value = 3.0,
                        step = 0.01)


# In[39]:


# Mapping Feature Labels with Slider Values

features = {
  'ph':ph,
  'Hardness':Hardness,
  'Solids':Solids,
  'Chloramines':Chloramines,
  'Sulfate':Sulfate,
  'Conductivity':Conductivity,
  'Organic_carbon':Organic_carbon,
  'Trihalomethanes':Trihalomethanes,
  'Turbidity':Turbidity}


# In[40]:


# Converting Features into DataFrame

features_df  = pd.DataFrame([features])

st.table(features_df)


# In[49]:


# Predicting Star Rating

if st.button('Predict'):
    water = ""
    prediction = predict_rating(model, features_df)
    if prediction == 0:
        water = "Não potável" 
    else:
        water = "Potável"
    
    st.write('Based on feature values, the classification of water potability is '+ water)


# In[ ]:




