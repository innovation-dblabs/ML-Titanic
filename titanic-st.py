# For creating the web app
import streamlit as st

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time

# SciKit Learn For ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# use altair for charting
import altair as alt

# Python Image Library
from PIL import Image

st.header("Data Science on the Titanic --- SPoOoOOOOOkY")


st.write( "Note: Sections in light gray like the one below - show the Python code used for this app.")
# echo command prints and executes the code.
with st.echo():
    # Load and display the Titanic image
    image = Image.open('Stöwer_Titanic.jpg')
    st.image(image, caption='Sinking of the titianic - courtesy Wikipedia', use_column_width=True)

# Streamlit works with markup syntax to automatically print this on the apps page. when you use the 3 apostrophes
'''

https://www.kaggle.com/c/titanic/overview

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

**Lets take a closer look at the data and see if we can idenitfy reasons that might have given some people a higher chance of survival.**
'''
# data files are in the same directory
DATA_URL="train.csv"
TEST_URL="test.csv"

def load_data():
    data = pd.read_csv(DATA_URL)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Function to impute the value of the missing Age to the average in that class
def add_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        return int(train[train["Pclass"] == Pclass]["Age"].mean())
    else:
        return Age


# echo command prints and executes the code.
with st.echo():
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    train = load_data()  # this function loads data from csv into a DataFrame
    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')

    # write out the df as a table
    num_records = train.shape
    st.write('### Dataframe loaded from the CSV file')
    st.write( train )
    st.write('### Total number of records (rows, cols) = ', num_records)

    st.write('### Data Summary')
    st.write(train.describe())

## st.subheader('Number of passengers by age')
## '''Use NumPy to generate a histogram. Bucket passengers by Age (0-100 across 10 bins):'''
## hist_values = np.histogram(
##     train['Age'], bins=10, range=(0,100))[0]
## st.bar_chart(hist_values)

'''
### Data Dictionary
https://www.kaggle.com/c/titanic/data

The above URL contains a description of all the data columns.
'''


'''
## Exploratory Data Analysis (EDA)
### A. Lets take a look at survivors by sex
'''
# echo command prints and executes the code.
with st.echo():
    c0 = alt.Chart(train).mark_bar().encode(
        x='Survived:N', y='count(PassengerId):Q', column='Sex:N', color='Survived'
    ).properties(
            title='Distribution of survivors (0=dead, 1=survived) by sex'
    )
    st.altair_chart(c0, width=100)

''' Observation: A lot more women survived compared to men'''


'''### B. Lets take a look at the distribution of people, by age, fare and survival'''
# echo command prints and executes the code.
with st.echo():
    c1 = alt.Chart(train).mark_circle().encode(
        x='Age', y='Fare', color='Survived'
    ).properties(
            title='Distribution of passengers, by age, fare and survival'
    )
    st.altair_chart(c1, width=800)

'''
### C. Passengers by sex ( ie male vs female ) across class of ticket (Pclass)

pclass: A proxy for socio-economic status (SES)
1st = Upper, 2nd = Middle, 3rd = Lower

'''

# echo command prints and executes the code.
with st.echo():
    c2 = alt.Chart(train).mark_bar().encode(
        x='Sex:N',
        y='count(PassengerId):Q',
        color='Survived:N',
        column='Pclass:N'
    ).properties(
        title='Passengers by sex across class of ticket (Pclass)'
    )
    st.altair_chart(c2, width=100)

'''
Observation:
We can infer that, as Molly, if you were a female and you were in class 1, probably you would survive.

On the other hand, if you were a man and you were in class 3, you didn’t have good chances to live.



## Machine Learning using Regression Analysis
Let's get the data ready for training a Logistic Regression Model.

'''

# echo command prints and executes the code.
with st.echo():
    # As we saw before, we have some null values for age.
    # Let’s create a function to impute ages to the corresponding age average per class.
    # Impute null age to the averaqe across that class of cabin
    train["Age"] = train[["Age", "Pclass"]].apply(add_age,axis=1)

    # We have lots of null values for Cabin column, so we just remove it.
    train.drop("Cabin",inplace=True,axis=1)

    # finally remove rows with null values
    train.dropna(inplace=True)

    # We are going to convert some categorical data into numeric. For example, the sex column.
    pd.get_dummies(train["Sex"])
    sex = pd.get_dummies(train["Sex"],drop_first=True)
    embarked = pd.get_dummies(train["Embarked"],drop_first=True)
    pclass = pd.get_dummies(train["Pclass"],drop_first=True)

    # add the above columns to the data set
    train = pd.concat([train,pclass,sex,embarked],axis=1)

    # drop columns we are not going to use
    train.drop(["PassengerId","Pclass","Name","Sex","Ticket","Embarked"],axis=1,inplace=True)

'''
Now our dataset is ready for the model.
X will contain all the features and y will contain the target variable.
'''
# prepare the training data X and y
X = train.drop("Survived",axis=1)
y = train["Survived"]

st.write( X )



'''
### Let’s use Logistic Regression to train the model:
We will use train_test_split from cross_validation module to split our data.
70% of the data will be training data and %30 will be testing data.
'''
# echo command prints and executes the code.
with st.echo():

    model_train_state = st.text('Training model...')
    # from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    model_train_state = st.text('Training complete.')


    '''
    ## Let’s see how accurate is our model for predictions:
    Lets see how the model performs when prediciting survival using the test data.
    '''

    predictions = logmodel.predict(X_test)
    st.write("### Classification Report")
    st.write(classification_report(y_test, predictions))
    st.write("### Confusion Matrix")
    st.write( confusion_matrix(y_test, predictions))

    '''
    - True positive: 149 (We predicted a positive result and it was positive)
    - True negative: 68 (We predicted a negative result and it was negative)
    - False positive: 14 (We predicted a positive result and it was negative)
    - False negative: 36 (We predicted a negative result and it was positive)

    We still can improve our model, this tutorial is intended to show how we can do some exploratory analysis,
    clean up data, perform predictions and talk about this event and this wonderful movie.

    '''


'''
## Lets take our trained model or a spin

Based on the above training, our model can predict with an accuracy of 81% if a passenger would have survived the Titanic given the following criteria
- Age  -- a Number
- SibSp  --  Number of sibling you are trabvelling with
- Parch  -- Family reslation rtravelling with ( parents, children )
- Fare  -- Passenger fare
- Pclass  -- A proxy for socio-economic status (SES)  [ 1|2|3 ]
- Male/Female  -- [ male | female ]
- Embarked from Queenstown -- [ 0|1 ]
- Embarked from Southampton -- [ 0|1 ]

Use one of the passengers from the test data below that the model was not trained on to check its prediction.
**Note:** This is data that the model has not seen before.

'''
# Load only 20 rows
test = pd.read_csv(TEST_URL)
st.write( test )
'''
Values have been populated from passengerId 892 below
'''

passengerAge = st.number_input("Passenger Age", value=34, min_value=0, max_value=80)
passengerSibSp = st.number_input("Passenger SibSp", value=0, min_value=0, max_value=8)
passengerParch = st.number_input("Passenger Parch", value=0, min_value=0, max_value=6)
passengerFare = st.number_input("Passenger Fare", value=8, min_value=0, max_value=520)
passengerPclass = st.number_input("Passenger Pclass", value=3, min_value=1, max_value=3)
if (passengerPclass == 2 ): passengerPclass2=1
else: passengerPclass2=0
if (passengerPclass == 3 ): passengerPclass3=1
else: passengerPclass3=0

if (st.selectbox("Passenger Sex", ('male','female')) == "male"): passengerSex=1
else: passengerSex=0
if(st.checkbox("Passenger Embarked from Q", value=True)): passengerEmbarkedQ=1
else: passengerEmbarkedQ=0
if(st.checkbox("Passenger Embarked from S", value=False)): passengerEmbarkedS=1
else: passengerEmbarkedS=0


inputData = [[ passengerAge, passengerSibSp, passengerParch, passengerFare, passengerPclass2, passengerPclass3, passengerSex, passengerEmbarkedQ, passengerEmbarkedS ]]

X_testPassenger = pd.DataFrame (inputData)

st.write ( "Input for this passenger", X_testPassenger )
st.write ( "**Model Survial Prediction (0=died, 1=survived)** ", logmodel.predict(X_testPassenger))
