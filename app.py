import altair as alt
import copy
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample

# Classification_report function will be used to check the working of the model
# Confusion matrix will be used for the confusion analysis of the model
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, plot_roc_curve, roc_curve, roc_auc_score


# Now we will define function to get train the model and return the model summary and performance

def Model_result_lin(model, data, dependent_col, independent_cols):
  # Create the input and output data
  X, y = data[independent_cols], data[dependent_col]
  # Now we will split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


  # Fit the logistic regression model
  model.fit(X_train, y_train)

  # Lets have a look at the results of the logistic regression for Graduation

  st.write('\nClassification report')
  pred = model.predict(X_test)
  st.write(classification_report(y_test, pred))

  prob_pred_test = model.predict_proba(X_test)[:, 1]
  prob_pred_train = model.predict_proba(X_train)[:, 1]

  st.write('\n\n\nThe ROC AUC score is shown below')
  st.write(roc_auc_score(y_test, prob_pred_test))

  st.write('\n\n\nThe following matrix show the confusion matrix')
  st.write('\n Confusion matrix shows the count of the true positive, false positive, true negative and false negative values')
  # fig, ax = plot_confusion_matrix(model, X_test, y_test)
  # st.pyplot(fig)
  st.write(confusion_matrix(y_test, pred))

  # plt.figure(figsize = (12, 5))
  # fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 5), dpi = 200)
  ns_fpr, ns_tpr, _ = roc_curve(y_test, prob_pred_test)
  lr_fpr, lr_tpr, _ = roc_curve(y_train, prob_pred_train)
  st.write('\n\n\nLets look at the roc curve as well')
  # axes[0].plot(ns_fpr, ns_tpr, label='Test', color = 'red', marker = 'v', mec = 'black')
  # axes[0].set_xlabel('False Positive Rate')
  # axes[0].set_ylabel('True Positive Rate')
  # axes[0].legend()
  # axes[1].plot(lr_fpr, lr_tpr, marker = 'v', label='Train', color = 'green', mec = 'black')
  # axes[1].set_xlabel('False Positive Rate')
  # axes[1].set_ylabel('True Positive Rate')
  # axes[1].legend()
  # st.pyplot(fig)



  fig = make_subplots(rows=1, cols=2)
  fig.add_trace(
      go.Line(x = ns_fpr, y = ns_tpr, name = 'Test'),
      row=1, col=1
  )

  fig.add_trace(
      go.Line(x = lr_fpr, y = lr_tpr, name = 'Train'),
      row=1, col=2
  )

  fig.update_layout(title_text="ROC curve",
    xaxis_title = 'False Positive Rate',
    yaxis_title = 'True Positive Rate')
  st.plotly_chart(fig, use_container_width=True)

  st.write('\nCoefficient Table')
  st.write(pd.DataFrame(model.coef_[0], index = independent_cols, columns = ['Coefficients']))
  st.write(f"The model in the above has an overall accuracy of {model.score(X_test, y_test)}. The goodness of the model can be verified by the roc_auc_score which is {roc_auc_score(y_test, prob_pred_test)}")


alt.renderers.set_embed_options(scaleFactor=2)



## Basic setup and app layout
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
st.title("House Hold Graduation")
st.sidebar.title("Control Panel")
left_col, middle_col, right_col = st.beta_columns(3)
palette = sns.color_palette("bright")

tick_size = 12
axis_title_size = 16


@st.cache
def generate_data():
    data = pd.read_csv('mst_new.csv')
    # data = pd.read_csv('mst1.csv')

    return data

# rng = np.random.default_rng(2)
# num_days = 14
data = generate_data()

# st.write(data)
data_transformed = pd.concat([data, pd.get_dummies(data['Program'])], axis = 1)
data_transformed.drop('Program', inplace = True, axis = 1)

df_majority = data_transformed[data_transformed.Graduated == 0]
df_minority = data_transformed[data_transformed.Graduated == 1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=12246,
                                 random_state=120)

data_final = pd.concat([df_majority, df_minority_upsampled])
# st.write(data_transformed.Graduated.value_counts())
## User inputs on the control panel
mode = st.sidebar.radio("Select Section: ", ('Data', 'Visualization', 'Model'))
if mode == 'Data':
    st.markdown("## Data")
    data_ver = st.sidebar.selectbox(
         'Knowing the data',
         ('Original Data', 'Number of null values', 'Transformed Data'))

    if data_ver == 'Original Data':
        st.write(data)
    elif data_ver == 'Number of null values':
        st.write(data.isna().sum())
    else:
        st.write(data_transformed)


elif mode == 'Visualization':
    st.markdown("## Visualization")
    data_look = st.sidebar.selectbox(
         'Visualizations',
         ('Count of graduated',
         'Housing cost vs HH income wrt Graduated',
         'Housing cost vs HH income wrt Gentrification',
         'Housing cost vs HH income wrt Program', ))

    if data_look == 'Count of graduated':
        df = data.groupby(by="Graduated").size().reset_index(name="counts")
        fig = px.bar(data_frame=df, x="Graduated", y="counts", color="Graduated", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
        st.write("The data can be seen unbalanced, the graduated people are in mejority (12246) and the non graduated people are in minority we might need to balance the dataset for the model building. The minority columns will be upscaled randomly.")
    elif data_look == 'Housing cost vs HH income wrt Graduated':
        # fig, ax = plt.subplots(figsize=(7, 3))
        # sns.lineplot(x = data['Housing_Cost'], y = data['HH_Income'], hue = data['Graduated'], ax = ax)
        # st.pyplot(fig)


        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Line(x = data[data.Graduated == 0].sort_values('Housing_Cost')['Housing_Cost'], y = data[data.Graduated == 0].sort_values('Housing_Cost')['HH_Income'], name = 'Graduated 0'),
            row=1, col=1
        )

        fig.add_trace(
            go.Line(x = data[data.Graduated == 1].sort_values('Housing_Cost')['Housing_Cost'], y = data[data.Graduated == 1].sort_values('Housing_Cost')['HH_Income'], name = 'Graduated 1'),
            row=1, col=1
        )

        fig.update_layout(title_text="Housing Cost vs HouseHold Income",
          xaxis_title = 'Housing_Cost',
          yaxis_title = 'HH_Income')
        st.plotly_chart(fig, use_container_width=True)
        st.write("The above graph shows the variation of the HH_income versus Housing_Cost, the clearly shows that the Housing cost is totally different for Graduated = 0 and Graduated = 1 people. The Cost is significantly low for Graduated = 1")
    elif data_look == 'Housing cost vs HH income wrt Gentrification':
        # fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (12, 21))
        # i = 0
        # color = ('red', 'blue', 'green', 'pink', 'yellow')
        # for val in data['Gentrification'].unique():
        #   data1 = data[data.Gentrification == val]
        #   sns.lineplot(x = data1['Housing_Cost'], y = data1['HH_Income'], hue = data1['Gentrification'], ax = axes[i], palette=[palette[i]])
        #   i += 1
        # fig.tight_layout()
        # st.pyplot(fig)

        fig = make_subplots(rows=5, cols=1)
        i = 1
        for val in data['Gentrification'].unique():
            data1 = data[data.Gentrification == val]
            fig.add_trace(
                go.Line(x = data1.sort_values('Housing_Cost')['Housing_Cost'].values,
                y = data1.sort_values('Housing_Cost')['HH_Income'].values,
                name = 'Gentrification : ' + str(val)),
                row=i, col=1
            )
            i = i + 1

        fig.update_layout(title_text="Housing Cost vs HouseHold Income",
          xaxis_title = 'Housing_Cost',
          yaxis_title = 'HH_Income',
          height = 1000)
        st.plotly_chart(fig, use_container_width=True)
        st.write("The above plot shows the plot of Housing_Cost versus HH_Income with a hue of Gentrification. The plot clearly shows that the graduated and non-graduated people are most clearly separable when the Gentrification is 5.")
    elif data_look == 'Housing cost vs HH income wrt Program':
        # fig, axes = plt.subplots(nrows = data['Program'].unique().shape[0], ncols = 1, figsize = (12, 30))
        # i = 0
        # color = ('red', 'blue', 'green', 'pink', 'yellow')
        # for val in data['Program'].unique():
        #   data1 = data[data.Program == val]
        #   sns.lineplot(x = data1['Housing_Cost'], y = data1['HH_Income'], hue = data1['Program'], ax = axes[i], palette=[palette[i%4]])
        #   i += 1
        # st.pyplot(fig)

        fig = make_subplots(rows=data['Program'].unique().shape[0], cols=1)
        i = 1
        for val in data['Program'].unique():
            data1 = data[data.Program == val]
            fig.add_trace(
                go.Line(x = data1.sort_values('Housing_Cost')['Housing_Cost'].values,
                y = data1.sort_values('Housing_Cost')['HH_Income'].values,
                name = 'Program : ' + str(val)),
                row=i, col=1
            )
            i = i + 1

        fig.update_layout(title_text="Housing Cost vs HouseHold Income",
          xaxis_title = 'Housing_Cost',
          yaxis_title = 'HH_Income',
          height = 4000)
        st.plotly_chart(fig, use_container_width=True)


# st.write(data_final.Graduated.value_counts())

# model = st.sidebar.selectbox(
#      'Select the model',
#      ('Logistic Regression',
#       'Random Forest Classifier',
#        'KNN Classifier'))

elif mode == 'Model':
    st.markdown("## Model Validation")
    columns = ['HH_Composition', 'HH_Income', 'Housing_Cost',
         'Gentrification', 'Education_TrackY', 'AHA-Owned', 'CRSHP',
         'ENH', 'FLOW', 'FUPF', 'FUPY', 'GAHVP', 'HFV', 'HOPE6', 'Homeflex',
         'ICPSH', 'JUNP10', 'MS1 (NED-PRE2008)', 'MS5', 'MS5-PFH',
         'Mixed - Homeflex', 'Mixed - PH', 'Mixed - RAD', 'NED', 'PAV', 'PHRR',
         'RADPH', 'REG', 'RISE II', 'SPHVS', 'VASH']
    vars_selected = st.sidebar.multiselect('Select Variables', columns)
    if vars_selected == []:
        vars_selected = columns
        lr = LogisticRegression()
        Model_result_lin(model = lr, data = data_final, dependent_col = 'Graduated', independent_cols = vars_selected)
    else:
        lr = LogisticRegression()
        Model_result_lin(model = lr, data = data_final, dependent_col = 'Graduated', independent_cols = vars_selected)
