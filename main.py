import pandas as pd
from distribution import check_distribution
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor
from normalization import normalize_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# READ :THE DATASET

data = pd.read_csv(r"C:/Users/CL/Desktop/excerise.csv")

# CHECK THE SHAPE
print(data.shape)

# DISPLAY THE DATA
print(data.head())

# CHECK THE DTYPES
print(data.dtypes)

# CHECK FOR NULL VALUES
print(data.isnull().sum())

# CHECK FOR MATHEMATICAL
print(data.describe())

# CHECK FOR DUBLICATED
print(data.duplicated())

# CHECK FOR CORELATION
print(data.corr()['maas'])


# CHECK THE DISTRIBUTION
check_distribution(data)
print(data)

# NORMALIZ THE DATA
n_data=normalize_data(data)

X_train,X_test,Y_train,Y_test = train_test_split(n_data.iloc[:,0:1],n_data.iloc[:,-1],test_size=0.2,random_state=2)


lr = Ridge(alpha=0.0001)

lr.fit(X_train,Y_train)

y_pred = lr.predict(X_test)

score = r2_score(Y_test,y_pred)

print(score)