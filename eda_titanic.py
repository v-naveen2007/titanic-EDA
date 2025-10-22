import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("titanic.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


df.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Numeric Columns")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.title("Boxplots of Numeric Features")
plt.show()

plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # keep only numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Columns Only)")
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

print("\nAverage Fare by Class:")
print(df.groupby('Pclass')['Fare'].mean())

print("\nSurvival Rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())