import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/students.csv')

# Step 1: Separate column types
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Step 2: Impute numerical columns (using mean)
numerical_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

# Step 3: Impute categorical columns (using most frequent)
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# ---- Feature Engineering ----

# Health Score: Sleep + Nutrition + Physical Activity
df['Health_Score'] = (
    df['Sleep_Patterns'] +
    df['Nutrition'].map({'Unhealthy': 0, 'Balanced': 1, 'Healthy': 2}) +
    df['Physical_Activity'].map({'Low': 0, 'Medium': 1, 'High': 2})
)

df['Distraction_Score'] = (
    df['Time_Wasted_on_Social_Media'] + 
    df['Sports_Participation'].map({'Low': 0, 'Medium': 1, 'High': 2}) + 
    df['Lack_of_Interest'].map({'Low': 0, 'Medium': 1, 'High': 2})
)

df['Support_Index'] = (
    df['Parental_Involvement'].map({'Low': 0, 'Medium': 1, 'High': 2}) +
    df['Tutoring'].map({'Yes': 1, 'No': 0}) +
    df['Mentoring'].map({'Yes': 1, 'No': 0})
)

# Study Engagement: Class Participation + Attendance + Study Hours
df['Attendance_Level'] = pd.cut(df['Attendance'], bins=[0, 80, 95, 100], labels=['Low', 'Medium', 'High'])

# Map 'Attendance_Level' to numeric values
df['Attendance_Level'] = df['Attendance_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Normalize Attendance to be on the same scale as Study_Hours and Class_Participation
attendance_scaler = StandardScaler()
df['Normalized_Attendance'] = attendance_scaler.fit_transform(df[['Attendance']])

# Combine the scaled Attendance with Study_Hours and Class_Participation
df['Study_Engagement'] = (
    df['Class_Participation'].map({'Low': 0, 'Medium': 1, 'High': 2}) + 
    df['Normalized_Attendance'] + 
    df['Study_Hours']
)
## check for missing vals
print(df.isnull().sum())

# ---- Handle Grades Column (if Grades is categorical) ----
# Map 'Grades' column to numeric values (if it's a categorical target)
df['Grades'] = df['Grades'].map({'A': 2, 'B': 1, 'C': 0})

# ---- Encoding Categorical Columns ----
# Separate Grades column from categorical columns before one-hot encoding
categorical_cols = [col for col in categorical_cols if col != 'Grades']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ---- Feature Scaling ----
# Scale selected numerical features, including engineered features
features_to_scale = ['Study_Hours', 'Screen_Time', 'Time_Wasted_on_Social_Media', 
                     'Class_Size', 'Health_Score', 'Distraction_Score', 'Support_Index', 'Study_Engagement']

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
