import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1️⃣ Load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("🔷 Raw data:")
print(df.head())

print("\n🔷 Missing values before:")
print(df.isnull().sum())

# 2️⃣ Fill missing values
# Age: fill with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Embarked: fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Cabin: drop (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

print("\n✅ Missing values after:")
print(df.isnull().sum())

# 3️⃣ Encode categorical variables
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])  # male=1, female=0

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# 4️⃣ Optional: remove outliers in Age (e.g., age > 100)
df = df[df['Age'] <= 100]

# 5️⃣ Drop irrelevant columns (Name, Ticket)
df.drop(columns=['Name', 'Ticket'], inplace=True)

print("\n🔷 Cleaned data sample:")
print(df.head())

# 6️⃣ Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_titanic.csv'")
