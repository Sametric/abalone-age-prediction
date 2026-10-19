import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import dump



df = pd.read_csv('Dataset/abalone.data.csv')
# df.info()
# print(df.isnull().sum())

# df['Sex'] = df['Sex'].map({
#     'M' : 1,
#     'F' : 2,
#     'I' : 3
# })
df = pd.get_dummies(df, columns=['Sex'], dtype=int)

df['Age'] = df['Rings'] + 1.5

X = df.drop(['Rings','Age'],axis=1)
# print(X)
y = df['Age']
# print(y)


X_train,X_test,y_train,y_test = train_test_split(
                                X,
                                y,
                                test_size=0.2,
                                random_state=42
)

model = LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)


dump(model,'Abalone_Age_Predictor.joblib')

# # Enter the abalone's physical measurements
# sex = input("Enter sex (M, F, I): ").upper()
# length = float(input("Enter length: "))
# diameter = float(input("Enter diameter: "))
# height = float(input("Enter height: "))
# whole_weight = float(input("Enter whole weight: "))
# shucked_weight = float(input("Enter shucked weight: "))
# viscera_weight = float(input("Enter viscera weight: "))
# shell_weight = float(input("Enter shell weight: "))
#
# # Convert the input into the format used during training
# abalone = pd.DataFrame([{
#     'Length': length,
#     'Diameter': diameter,
#     'Height': height,
#     'Whole weight': whole_weight,
#     'Shucked weight': shucked_weight,
#     'Viscera weight': viscera_weight,
#     'Shell weight': shell_weight,
#     'Sex_F': int(sex == 'F'),
#     'Sex_I': int(sex == 'I'),
#     'Sex_M': int(sex == 'M')
# }])
#
# # Ensure the feature order matches the training data
# abalone = abalone[X.columns]
#
# # Predict age
# predicted_age = model.predict(abalone)[0]
#
# print(f"Predicted age: {predicted_age:.2f} years")





# sns.heatmap(df.corr(),annot=True)
# plt.show()
# print(df.to_string())
# print(df.describe())

