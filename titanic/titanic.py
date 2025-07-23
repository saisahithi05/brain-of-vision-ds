import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 📥 Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 🔷 Basic cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

df = df[df['Age'] <= 100]  # remove outliers

print("\n✅ Data cleaned!")

# --------------------------------------------
# 📊 DESCRIPTIVE ANALYSIS
print("\n🔷 DESCRIPTIVE ANALYSIS")
print(df.describe(include='all'))
print("\nCorrelation matrix:\n", df.corr())

print("\nSurvival rate:")
print(df['Survived'].value_counts(normalize=True))

# --------------------------------------------
# 📈 EDA
print("\n🔷 EXPLORATORY DATA ANALYSIS (EDA)")

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# --------------------------------------------
# 🔮 PREDICTIVE ANALYSIS
print("\n🔷 PREDICTIVE ANALYSIS")

X = df.drop(columns=['Survived', 'PassengerId'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------------------------------
# 🧭 PRESCRIPTIVE ANALYSIS
print("\n🔷 PRESCRIPTIVE ANALYSIS")

pred_probs = model.predict_proba(X_test)[:,1]
recommendations = []

for prob in pred_probs:
    if prob >= 0.7:
        recommendations.append("Prioritize safety")
    elif prob >= 0.4:
        recommendations.append("Monitor closely")
    else:
        recommendations.append("Low risk")

prescriptive_df = X_test.copy()
prescriptive_df['Survival_Prob'] = pred_probs
prescriptive_df['Recommendation'] = recommendations

print(prescriptive_df.head())

# Save prescriptive suggestions
prescriptive_df.to_csv("prescriptive_titanic.csv", index=False)
print("\n✅ Prescriptive recommendations saved as 'prescriptive_titanic.csv'")
