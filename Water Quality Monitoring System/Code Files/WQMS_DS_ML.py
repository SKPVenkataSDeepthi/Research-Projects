#!/usr/bin/env python
# coding: utf-8

# # Part1: Cleaning, wrangling data & Exploratory Data Analysis (EDA)

# Data cleaning focuses on removing inaccurate data from your data set whereas data wrangling focuses on transforming the data's format, typically by converting “raw” data into another format more suitable for use. 

# In[2]:


Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.

Here are the steps for EDA:

***Steps in EDA***:
1. Provide descriptions of your sample and features
2. Check for missing data
3. Identify the shape of your data
4. Identify significant correlations
5. Spot/deal with outliers in the dataset



# # xyz

# In[ ]:





# # 1.1 Provide descriptions of your sample and features

# In[1]:


#Read in libraries
import numpy as np
from sklearn import preprocessing
import pandas as pd
 


# In[63]:


#Read the csv file
import pandas as pd

# Specify the file path
file_path = "WaterQltySys.csv"

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("File read successfully.")
    print(df.head())  # Display the first few rows of the DataFrame
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except Exception as e:
    print("An error occurred:", e)


# In[64]:


#Counting the number of rows and columns
rows = len(df.axes[0])
cols = len(df.axes[1])
print("Number of Rows: " + str(rows))
print("Number of Columns: " + str(cols))


# In[65]:


#Representing the datatypes
df.info()


# In[66]:


#Drop null values
df1=df.dropna(axis=1)
df1


# In[67]:


df1.info()
df1.head()


# In[68]:


df['created_at'] = pd.to_datetime(df['created_at'])


# In[69]:


df.info()


# # 1.2 Check for missing data

# In[70]:


#check for missing data
df.isnull().sum()


# In[71]:


#checking for duplicate values
df.duplicated().sum()


# # 1.3 Identify the shape of your data

# In[72]:


#Identifying shape of data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#countplot
sns.countplot(x='pH', data=df, )


# In[73]:


#pairplot
sns.pairplot(df, hue="pH",height=3)


# In[74]:


#histogram
import matplotlib.pyplot as plt

df['pH'].hist(bins=20, edgecolor='black')

plt.title('Distribution of pH Values')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.show()


# # 1.4 Identify significant correlations
# 

# In[75]:


corr = df.corr(method='spearman')
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# # 1.5 Spot/deal with outliers in the dataset

# In[76]:


print(df.columns)
df.info()


# In[77]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


dfm = pd.melt(df, id_vars=["created_at", "entry_id"])

# Create the boxplot
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=dfm, x="variable", y="value", hue="variable", dodge=True, fliersize=10)

# Annotate outliers
for i, box in enumerate(ax.artists):
    # Get the outliers for each boxplot
    outliers = dfm[dfm['variable'] == box.get_label()]
    # Find outliers in the y-axis
    outliers_y = outliers['value'][outliers['value'] > box.get_capline().get_ydata()[1]]
    for y in outliers_y:
        ax.text(i, y, f'{y:.2f}', horizontalalignment='center', size='small', color='red', weight='semibold')

plt.title("Boxplot with Annotated Outliers")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()


# In[78]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Specify the file path
file_path = "WaterQltySys.csv"

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("File read successfully.")

    # Drop rows with missing values
    df = df.dropna()
    print("Rows with missing values dropped.")
    
    # Removing outliers using z-score
    numeric_cols = df.select_dtypes(include='number').columns
    z_scores = stats.zscore(df[numeric_cols])
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df = df[filtered_entries]
    print("Outliers removed.")
    
    # Displaying boxplot after removing outliers
    dfm = pd.melt(df, id_vars=["pH"], value_vars=numeric_cols.difference(["pH"]))
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=dfm, x="variable", y="value", hue="variable", dodge=True, fliersize=10)
    plt.title("Boxplot after Removing Outliers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()
    
    print(df.head())  # Display the DataFrame after removing outliers
    
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except Exception as e:
    print("An error occurred:", e)


# # Part 2: Multiple Regression Analysis

# Multiple regression analysis is a statistical technique used to understand the relationship between a dependent variable and two or more independent variables. 

# # 2.1 Multiple Regression

# In[79]:


from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd


# In[80]:


# Add a constant term to the independent variables (intercept)
df = sm.add_constant(df)


# In[81]:


# Define the dependent variable and the independent variables
Y=df['pH']
X=df[['entry_id', 'Tempareture','TDS','Turbidity', 'created_at']]
X=df.drop(columns='pH')
X


# In[82]:


import pandas as pd

# Check data types
print("Data Types:")
print("pH:", df['pH'].dtype)
print("Tempareture:", df['Tempareture'].dtype)
print("TDS:", df['TDS'].dtype)
print("Turbidity:", df['Turbidity'].dtype)


# In[83]:


import statsmodels.api as sm

# Define the dependent variable (target)
Y = df['pH']

# Define the independent variables (features)
X = df[['Tempareture', 'TDS', 'Turbidity']]

# Add a constant term to the independent variables (intercept)
X = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(Y, X).fit()

# Print the summary of the model
print(model.summary())


# In[84]:


# Filter significant variables from model1 based on a 90% confidence level (p-value <= 0.1)
significant_variables = model.pvalues[model.pvalues <= 0.1].index


# In[85]:


model2 = df[significant_variables]
model2['pH'] = df['pH']
Y_model2 = model2['pH']
X_model2 = model2.drop(columns=['pH'])
model2 = sm.OLS(Y_model2, X_model2).fit()
print(model2.summary())


# # 2.2 Compare the two models with ANOVA

# In[86]:


#Comparing two models by ANOVA
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm


# In[87]:


# Fit Model1 and Model2 as described in previous responses
model = sm.OLS(Y, X).fit()
model2 = sm.OLS(Y_model2, X_model2).fit()


# In[88]:


# Perform ANOVA to compare the two models
anova_results = anova_lm(model, model2)


# In[89]:


# Print the ANOVA table
print(anova_results)


# Comparing these metrics, Model 1 has a slightly higher R-squared and F-statistic, indicating that it explains a slightly greater proportion of the variance in the dependent variable (pH) and is a better fit overall. Additionally, Model 1 includes an extra variable (TDS), which might be valuable in explaining pH variability.
# 
# Therefore, based on these metrics, Model 1 appears to be the better choice.

# # 2.3 Checking assumptions

# In[90]:


#Checking Assumptions
import seaborn as sns

# Create a pair plot to visualize relationships
sns.pairplot(df, x_vars=['entry_id', 'Tempareture','TDS','Turbidity', 'created_at'], y_vars=['pH'])


# In[91]:


#Independence of the errors
from statsmodels.stats.stattools import durbin_watson

# Calculate the Durbin-Watson statistic
dw_statistic = durbin_watson(model.resid)
print(dw_statistic)


# # 2.4 Homoscedasticity

# In[92]:


#Homoscedasticity
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# Fit your multiple regression model using OLS
model = sm.OLS(endog=Y, exog=X).fit()

# Get the residuals
residuals = model.resid

# Perform statistical tests for homoscedasticity
het_test = sms.het_breuschpagan(residuals, model.model.exog)
white_test = sms.het_white(residuals, model.model.exog)

# Plot residuals vs. fitted values
plt.scatter(model.fittedvalues, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='r', linestyle='--')

plt.show()

# Print the test results
print("Breusch-Pagan test p-value:", het_test[1])
print("White test p-value:", white_test[1])


# In[93]:


import matplotlib.pyplot as plt

plt.scatter(model2.fittedvalues, residuals)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()


# # 2.5 Normality of Residuals

# In[94]:


#Normality of Residuals:
import statsmodels.api as sm
import scipy.stats as stats

# Create a histogram of residuals
sm.graphics.tsa.plot_acf(model.resid, lags=40)

# Create a Q-Q plot of residuals
stats.probplot(model.resid, dist="norm", plot=plt)

# Create a histogram of residuals
sm.graphics.tsa.plot_acf(model2.resid, lags=40)

# Create a Q-Q plot of residuals
stats.probplot(model2.resid, dist="norm", plot=plt)


# # 2.6 Multicollinearity

# In[95]:


#No or Low Multicollinearity:
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True)


# In[96]:


#No or Low Outliers:
# Create a scatterplot of residuals against predicted values
plt.scatter(model.fittedvalues, model.resid)

#No or Low Outliers:
# Create a scatterplot of residuals against predicted values
plt.scatter(model2.fittedvalues, model2.resid)


# # Part 3: Feature Selection

# There are three types of feature selection techniques. They were:
# 
# 1. Filter methods
# 2. Wrapper methods
# 3. Embedded methods

# # 3.1 Filter methods

# In[97]:


from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np


# Instantiate the feature selector
selector = SelectKBest(score_func=f_classif, k='all')

# Fit the selector to your data
fit = selector.fit(X, Y)

# Now we can access the scores and p-values
features_score = pd.DataFrame(fit.scores_)
features_pvalue = pd.DataFrame(np.round(fit.pvalues_, 4))
features = pd.DataFrame(X.columns)
feature_score = pd.concat([features, features_score, features_pvalue], axis=1)

# Assigning the column name
feature_score.columns = ["Input_Features", "F_Score", "P_Value"]
print(feature_score.nlargest(30, columns="F_Score"))


# In[98]:


#Feature Selection using Correlation Matrix with Heatmap (Filtered Method)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a correlation matrix
correlation_matrix = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap with seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Show the plot
plt.show()


# In[99]:


#Feature Slection using Mutual Information(MI) or Information Gain(IG) (Filtered Method)
from sklearn.feature_selection import mutual_info_regression
mir = mutual_info_regression(X,Y)
mrs_score = pd.Series(mir,index=X.columns)
mrs_score.sort_values(ascending=False)


# # 3.2 Wrapper methods

# In[100]:


#Recurssive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset for demonstration
X, Y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Create the RFE model and select 10 features
rfe = RFE(estimator=clf, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Fit the model on the reduced feature set
clf.fit(X_train_rfe, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_rfe)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with selected features: {accuracy}")

# Print the selected features
selected_features = [f"Feature {i+1}" for i in range(len(rfe.support_)) if rfe.support_[i]]
print("Selected Features:", selected_features)


# In[101]:


#Exhaustive feature selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector
import numpy as np

# Generate synthetic data for demonstration
np.random.seed(42)
X_synthetic = np.random.rand(100, 5)
y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 1).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# Create a RandomForestClassifier 
model = RandomForestClassifier(random_state=42)

# Create ExhaustiveFeatureSelector
efs = ExhaustiveFeatureSelector(model,
                                min_features=1,
                                max_features=X_train.shape[1],
                                scoring='accuracy',
                                print_progress=True,
                                cv=5)

# Fit the selector to the data
efs = efs.fit(X_train, y_train)

# Get the selected feature indices
selected_features = list(efs.best_idx_)

# Use the selected features for training and testing
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

# Train the model with selected features
model.fit(X_train_selected, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_selected)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
print("Selected features:", selected_features)
print("Accuracy on the test set:", accuracy)


# In[102]:


#Forward Selection
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.ensemble import RandomForestClassifier as rfc


# In[103]:


df = pd.DataFrame(columns=['entry_id', 'Tempareture','TDS','Turbidity', 'created_at','pH'])
columns = df.columns
feature_names=tuple(df.columns)
feature_names


# In[104]:


X.shape, Y.shape


# In[105]:


sfs1 = SFS(#knn(n_neighbors=3),
           #rfc(n_jobs=8),
           LGR(max_iter=1000),
           k_features='best', 
           forward=True, 
           floating=False, 
           verbose=2,
           #scoring = 'neg_mean_squared_error',  # sklearn regressors
           scoring='accuracy',  # sklearn classifiers
           cv=0)
sfs1 = sfs1.fit(X, Y,feature_names)


# In[106]:


sfs1.subsets_


# In[107]:


sfs1.get_metric_dict()


# In[108]:


sfs1.k_feature_names_, sfs1.k_feature_idx_


# In[109]:


df = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T
df[["feature_idx","avg_score"]]


# # 3.3 Embedded methods

# In[110]:


#LASSO

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
Y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100)  # Linear combination with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Lasso model
lasso = Lasso(alpha=0.01) 

# Fit the model to the training data
lasso.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients
print('Coefficients:', lasso.coef_)


# In[111]:


#RIDGE regression 

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
Y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100)  # Linear combination with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a Ridge model
ridge = Ridge(alpha=1.0)  

# Fit the model to the training data
ridge.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients
print('Coefficients:', ridge.coef_)


# In[112]:


#Elastic Net
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
Y = 2 * X[:, 0] + 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100)  # Linear combination with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create an Elastic Net model
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Fit the model to the training data
elastic_net.fit(X_train, y_train)

# Make predictions on the test set
y_pred = elastic_net.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Print the coefficients
print('Coefficients:', elastic_net.coef_)


# # Implementation of Machine Learning Algorithms

# # Enhancing Water Quality Monitoring: Integrating Neural Networks for Advanced Analysis

# Neural Network Library

# In[113]:


import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs
    
    def backward(self, gradients):
        # Calculating the gradients w.r.t. weights and biases
        weights_gradients = np.dot(self.inputs.T, gradients)
        biases_gradients = np.sum(gradients, axis=0, keepdims=True)

        # Updating the weights and biases
        self.weights -= learning_rate * weights_gradients
        self.biases -= learning_rate * biases_gradients

        # Calculating the gradients w.r.t. inputs 
        return np.dot(gradients, self.weights.T)

class Activation:
    def __init__(self, activation_func, activation_func_derivative):
        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
    
    def forward(self, inputs):
        self.inputs = inputs
        return self.activation_func(inputs)
    
    def backward(self, gradients):
        return gradients * self.activation_func_derivative(self.inputs)

class Loss:
    @staticmethod
    def mean_squared_error(predictions, targets):
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mean_squared_error_derivative(predictions, targets):
        return 2 * (predictions - targets) / len(predictions) 

class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def train(self, X_train, y_train, learning_rate, epochs):
        for epoch in range(epochs):
            predictions = self.forward(X_train)
            loss = Loss.mean_squared_error(predictions, y_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

            # Backpropagation
            error = Loss.mean_squared_error_derivative(predictions, y_train)
            for layer in reversed(self.layers):
                error = layer.backward(error)

# Defining the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))




# The Layer Class

# In[114]:


import numpy as np

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs
    
    def backward(self, gradients):
        weights_gradients = np.dot(self.inputs.T, gradients)
        biases_gradients = np.sum(gradients, axis=0, keepdims=True)
        input_gradients = np.dot(gradients, self.weights.T)
        
       
        
        return input_gradients


# Linear Layer

# In[115]:


import numpy as np

class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        pass
    
    def backward(self, gradients):
        pass

class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.inputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases
    
    def backward(self, gradients):
        weights_gradients = np.dot(self.inputs.T, gradients)
        biases_gradients = np.sum(gradients, axis=0, keepdims=True)
        input_gradients = np.dot(gradients, self.weights.T)
        
        
        
        return input_gradients, weights_gradients, biases_gradients


# Sigmoid Function

# In[116]:


import numpy as np

class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        pass
    
    def backward(self, gradients):
        pass

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    
    def backward(self, gradients):
        sigmoid_derivative = self.outputs * (1 - self.outputs)
        return gradients * sigmoid_derivative


# Rectified Linear Unit (ReLU)

# In[117]:


import numpy as np

class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        pass
    
    def backward(self, gradients):
        pass

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
        return self.outputs
    
    def backward(self, gradients):
        relu_derivative = np.where(self.inputs > 0, 1, 0)
        return gradients * relu_derivative


# Binary Cross-Entropy Loss

# In[118]:


import numpy as np

class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        pass
    
    def backward(self, gradients):
        pass

class BinaryCrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        self.inputs = predictions
        self.targets = targets
        return -np.mean(targets * np.log(predictions + 1e-15) + (1 - targets) * np.log(1 - predictions + 1e-15))
    
    def backward(self):
        return (self.inputs - self.targets) / (self.inputs * (1 - self.inputs))


# The Sequential Class

# In[119]:


class Sequential(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, gradients):
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
        return gradients


# Saving and Loading

# In[120]:


def save_weights(self, filename):
    weights = [layer.weights for layer in self.layers if hasattr(layer, 'weights')]
    biases = [layer.biases for layer in self.layers if hasattr(layer, 'biases')]
    np.savez(filename, weights=weights, biases=biases)

def load_weights(self, filename):
    data = np.load(filename)
    for layer, weights, biases in zip(self.layers, data['weights'], data['biases']):
        if hasattr(layer, 'weights'):
            layer.weights = weights
        if hasattr(layer, 'biases'):
            layer.biases = biases


# Testing the Library

# In[121]:


import numpy as np

# XOR input data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# XOR labels
y = np.array([[0],
              [1],
              [1],
              [0]])

class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = inputs  
        return inputs
    
    def backward(self, gradients):
        pass

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
        return self.outputs
    
    def backward(self, gradients):
        sigmoid_derivative = self.outputs * (1 - self.outputs)
        return gradients * sigmoid_derivative

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
        self.outputs = None
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.tanh(inputs)
        return self.outputs
    
    def backward(self, gradients):
        tanh_derivative = 1 - np.tanh(self.inputs)**2
        return gradients * tanh_derivative

# Defining the neural network 
class XOR_Model:
    def __init__(self):
        self.hidden_layer = Layer()
        self.output_layer = SigmoidLayer()
    
    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs.dot(self.hidden_weights) + self.hidden_bias)
        output = self.output_layer.forward(hidden_output.dot(self.output_weights) + self.output_bias)
        return output
    
    def backward(self, gradients):
        gradients = self.output_layer.backward(gradients)
        gradients = self.hidden_layer.backward(gradients.dot(self.output_weights.T))
        return gradients
    
    def train(self, X, y, learning_rate=0.1, epochs=10000):
        np.random.seed(0)
        self.hidden_weights = np.random.randn(X.shape[1], 2)
        self.hidden_bias = np.zeros((1, 2))
        self.output_weights = np.random.randn(2, 1)
        self.output_bias = np.zeros((1, 1))
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Computing the loss
            loss = np.mean((output - y) ** 2)
            
            # Backward pass
            gradient = 2 * (output - y) / X.shape[0]
            self.backward(gradient)
            
            # Updating the weights
            self.hidden_weights -= learning_rate * X.T.dot(self.hidden_layer.inputs)
            self.hidden_bias -= learning_rate * np.sum(self.hidden_layer.inputs, axis=0, keepdims=True)
            self.output_weights -= learning_rate * self.hidden_layer.outputs.T.dot(gradient)
            self.output_bias -= learning_rate * np.sum(gradient, axis=0, keepdims=True)
            
            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}")
 
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        print("\nThresholded Output Matrix:")
        print(output.astype(int))

# Training the model with sigmoid activations
print("Training with sigmoid activations:")
model_sigmoid = XOR_Model()
model_sigmoid.train(X, y)

# Saving the weights
np.savez('XOR_solved_sigmoid.npz', hidden_weights=model_sigmoid.hidden_weights,
                                    hidden_bias=model_sigmoid.hidden_bias,
                                    output_weights=model_sigmoid.output_weights,
                                    output_bias=model_sigmoid.output_bias)

# Training the model with hyperbolic tangent activations
print("\nTraining with hyperbolic tangent activations:")
class XOR_Model_Tanh(XOR_Model):
    def __init__(self):
        super().__init__()
        self.hidden_layer = TanhLayer()

model_tanh = XOR_Model_Tanh()
model_tanh.train(X, y)

# Saving weights
np.savez('XOR_solved_tanh.npz', hidden_weights=model_tanh.hidden_weights,
                                 hidden_bias=model_tanh.hidden_bias,
                                 output_weights=model_tanh.output_weights,
                                 output_bias=model_tanh.output_bias)


# # Model Selection

# In[122]:


import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, activation_function, activation_derivative):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        
        # Initializing the weights and biases
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.bias1 = np.zeros((1, hidden_dim))
        self.weights2 = np.random.randn(hidden_dim, hidden_dim)
        self.bias2 = np.zeros((1, hidden_dim))
        self.weights3 = np.random.randn(hidden_dim, output_dim)
        self.bias3 = np.zeros((1, output_dim))
        
    def forward(self, X):
        # Forward pass 
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.activation_function(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.activation_function(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        exp_scores = np.exp(self.z3)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y, learning_rate):
        # Backpropagation
        delta4 = self.probs
        delta4[range(len(X)), y] -= 1
        dW3 = np.dot(self.a2.T, delta4)
        db3 = np.sum(delta4, axis=0, keepdims=True)
        delta3 = np.dot(delta4, self.weights3.T) * self.activation_derivative(self.a2)
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0)
        delta2 = np.dot(delta3, self.weights2.T) * self.activation_derivative(self.a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # Updating the weights and biases
        self.weights1 -= learning_rate * dW1
        self.bias1 -= learning_rate * db1
        self.weights2 -= learning_rate * dW2
        self.bias2 -= learning_rate * db2
        self.weights3 -= learning_rate * dW3
        self.bias3 -= learning_rate * db3

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def train(X_train, y_train, X_val, y_val, num_epochs, learning_rate, activation_function, activation_derivative, model_name, batch_size=128):
    input_dim = X_train.shape[1]
    output_dim = np.max(y_train) + 1
    hidden_dim = 3 
    
    model = SimpleNeuralNetwork(input_dim, hidden_dim, output_dim, activation_function, activation_derivative)
    
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    train_losses = []
    val_losses = []
    
    num_train_samples = X_train.shape[0]
    
    for epoch in range(num_epochs):
        # Shuffling the training data
        permutation = np.random.permutation(num_train_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        
        for i in range(0, num_train_samples, batch_size):
            
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward pass
            probs = model.forward(X_batch)
            
            # Compute loss
            corect_logprobs = -np.log(probs[range(len(X_batch)), y_batch])
            data_loss = np.sum(corect_logprobs)
            loss = 1./len(X_batch) * data_loss
            
            # Backpropagation
            model.backward(X_batch, y_batch, learning_rate)
        
        # Forward pass on validation set
        probs_val = model.forward(X_val)
        
        # Computing validation loss
        corect_logprobs_val = -np.log(probs_val[range(len(X_val)), y_val])
        val_loss = np.sum(corect_logprobs_val)
        val_loss = 1./len(X_val) * val_loss
        
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        train_losses.append(loss)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count == 3:
                print("Early stopping!")
                break
    
    # Plotting the training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.show()
    
    return model


np.random.seed(0)
X_train = np.random.randn(1000, 2)
y_train = np.random.randint(0, 2, 1000)
X_val = np.random.randn(200, 2)
y_val = np.random.randint(0, 2, 200)


X_test = np.random.randn(200, 2)
y_test = np.random.randint(0, 2, 200)

# Experimenting with hyperparameters
learning_rates = [0.01, 0.001, 0.0001]
num_epochs = 100

activation_functions = [sigmoid, relu, linear]  
activation_derivatives = [sigmoid_derivative, relu_derivative, linear_derivative]  
model_names = ['Model 1', 'Model 2', 'Model 3']

for i, lr in enumerate(learning_rates):
    print(f"Training with learning rate: {lr}")
    trained_model = train(X_train, y_train, X_val, y_val, num_epochs, lr, activation_functions[i], activation_derivatives[i], model_names[i], batch_size=128)
    print("Evaluation on test set:")
    test_probs = trained_model.forward(X_test)
    test_predictions = np.argmax(test_probs, axis=1)
    
    # Calculating RMSLE
    rmsle_value = rmsle(y_test, test_predictions)
    print(f"RMSLE for {model_names[i]}: {rmsle_value:.4f}")


