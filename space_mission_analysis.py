# ============================================================
#             SPACE MISSION DATA ANALYSIS
#             Python College Project
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. CREATE SPACE MISSION DATASET
# ============================================================

data = {
    "Year": [
        2018, 2018, 2018,
        2019, 2019, 2019,
        2020, 2020, 2020,
        2021, 2021, 2021,
        2022, 2022, 2022,
        2023, 2023, 2023,
        2024, 2024
    ],

    "Country": [
        "India", "USA", "Russia",
        "India", "USA", "China",
        "USA", "India", "China",
        "USA", "India", "Russia",
        "USA", "India", "China",
        "India", "USA", "China",
        "India", "USA"
    ],

    "Organization": [
        "ISRO", "NASA", "Roscosmos",
        "ISRO", "SpaceX", "CNSA",
        "SpaceX", "ISRO", "CNSA",
        "NASA", "ISRO", "Roscosmos",
        "SpaceX", "ISRO", "CNSA",
        "ISRO", "SpaceX", "CNSA",
        "ISRO", "NASA"
    ],

    "Mission_Name": [
        "GSLV Mission",
        "Mars Mission",
        "Soyuz Mission",
        "Chandrayaan-2",
        "Starlink-1",
        "Chang'e-4",
        "Starlink-2",
        "Gaganyaan Test",
        "Long March",
        "Mars Perseverance",
        "PSLV Mission",
        "Soyuz MS",
        "Starlink-10",
        "SSLV-D1",
        "Chang'e-5",
        "Chandrayaan-3",
        "Crew Dragon",
        "Tiangong",
        "Aditya-L1",
        "Artemis Test"
    ],

    "Payload_kg": [
        2200, 3000, 2500,
        3850, 15000, 1200,
        14500, 2000, 4000,
        1025, 1800, 3000,
        16000, 500, 8200,
        3900, 12500, 4500,
        1500, 9000
    ],

    "Mission_Type": [
        "Satellite",
        "Mars",
        "Satellite",
        "Lunar",
        "Satellite",
        "Lunar",
        "Satellite",
        "Human Spaceflight",
        "Satellite",
        "Mars",
        "Satellite",
        "Human Spaceflight",
        "Satellite",
        "Satellite",
        "Lunar",
        "Lunar",
        "Human Spaceflight",
        "Space Station",
        "Solar",
        "Lunar"
    ],

    "Outcome": [
        "Success",
        "Success",
        "Success",
        "Failure",
        "Success",
        "Success",
        "Success",
        "Success",
        "Success",
        "Success",
        "Success",
        "Failure",
        "Success",
        "Failure",
        "Success",
        "Success",
        "Success",
        "Success",
        "Success",
        "Success"
    ]
}


df = pd.DataFrame(data)


# ============================================================
# 2. SAVE DATASET AS CSV
# ============================================================

df.to_csv("space_missions.csv", index=False)


# ============================================================
# 3. DISPLAY PROJECT TITLE
# ============================================================

print("\n")
print("=" * 60)
print("             SPACE MISSION DATA ANALYSIS")
print("=" * 60)


# ============================================================
# 4. DISPLAY DATASET
# ============================================================

print("\nFIRST 5 RECORDS")
print("-" * 60)
print(df.head())


# ============================================================
# 5. DATA INFORMATION
# ============================================================

print("\nDATASET INFORMATION")
print("-" * 60)

print("Total Missions:", len(df))
print("Total Columns:", len(df.columns))
print("Columns:", list(df.columns))


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\nMISSING VALUES")
print("-" * 60)

print(df.isnull().sum())


# ============================================================
# 7. CHECK DUPLICATE VALUES
# ============================================================

print("\nDUPLICATE ROWS")
print("-" * 60)

print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()


# ============================================================
# 8. BASIC ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("                 BASIC ANALYSIS")
print("=" * 60)


# Mission outcomes
print("\nMISSION OUTCOMES")
print("-" * 60)

outcomes = df["Outcome"].value_counts()
print(outcomes)


# Missions by country
print("\nMISSIONS BY COUNTRY")
print("-" * 60)

countries = df["Country"].value_counts()
print(countries)


# Missions by organization
print("\nMISSIONS BY ORGANIZATION")
print("-" * 60)

organizations = df["Organization"].value_counts()
print(organizations)


# Mission types
print("\nMISSION TYPES")
print("-" * 60)

mission_types = df["Mission_Type"].value_counts()
print(mission_types)


# ============================================================
# 9. SUCCESS RATE
# ============================================================

total_missions = len(df)

successful_missions = (
    df["Outcome"] == "Success"
).sum()

failed_missions = (
    df["Outcome"] == "Failure"
).sum()

success_rate = (
    successful_missions / total_missions
) * 100


print("\n")
print("=" * 60)
print("                 SUCCESS ANALYSIS")
print("=" * 60)

print("Total Missions      :", total_missions)
print("Successful Missions :", successful_missions)
print("Failed Missions     :", failed_missions)
print("Success Rate        :", round(success_rate, 2), "%")


# ============================================================
# 10. MOST ACTIVE COUNTRY
# ============================================================

most_active_country = countries.idxmax()
most_active_country_count = countries.max()

print("\nMost Active Country:")
print(
    most_active_country,
    "-",
    most_active_country_count,
    "missions"
)


# ============================================================
# 11. MOST ACTIVE ORGANIZATION
# ============================================================

most_active_org = organizations.idxmax()
most_active_org_count = organizations.max()

print("\nMost Active Organization:")
print(
    most_active_org,
    "-",
    most_active_org_count,
    "missions"
)


# ============================================================
# 12. ALL 5 GRAPHS TOGETHER
# ============================================================

plt.figure(figsize=(14, 10))


# ------------------------------------------------------------
# GRAPH 1: Missions by Year
# ------------------------------------------------------------

plt.subplot(3, 2, 1)

year_count = df["Year"].value_counts().sort_index()

plt.bar(
    year_count.index,
    year_count.values
)

plt.title("Space Missions by Year")
plt.xlabel("Year")
plt.ylabel("Number of Missions")


# ------------------------------------------------------------
# GRAPH 2: Success vs Failure
# ------------------------------------------------------------

plt.subplot(3, 2, 2)

outcome_count = df["Outcome"].value_counts()

plt.pie(
    outcome_count.values,
    labels=outcome_count.index,
    autopct="%1.1f%%"
)

plt.title("Mission Success vs Failure")


# ------------------------------------------------------------
# GRAPH 3: Missions by Country
# ------------------------------------------------------------

plt.subplot(3, 2, 3)

country_count = df["Country"].value_counts()

plt.bar(
    country_count.index,
    country_count.values
)

plt.title("Missions by Country")
plt.xlabel("Country")
plt.ylabel("Number of Missions")

plt.xticks(rotation=45)


# ------------------------------------------------------------
# GRAPH 4: Missions by Organization
# ------------------------------------------------------------

plt.subplot(3, 2, 4)

org_count = df["Organization"].value_counts()

plt.bar(
    org_count.index,
    org_count.values
)

plt.title("Missions by Organization")
plt.xlabel("Organization")
plt.ylabel("Number of Missions")

plt.xticks(rotation=45)


# ------------------------------------------------------------
# GRAPH 5: Mission Types
# ------------------------------------------------------------

plt.subplot(3, 2, 5)

type_count = df["Mission_Type"].value_counts()

plt.bar(
    type_count.index,
    type_count.values
)

plt.title("Mission Types")
plt.xlabel("Mission Type")
plt.ylabel("Number of Missions")

plt.xticks(rotation=45)


# Adjust graph spacing
plt.tight_layout()

# Display all 5 graphs
plt.show()


# ============================================================
# 13. MACHINE LEARNING
# ============================================================

print("\n")
print("=" * 60)
print("             MACHINE LEARNING")
print("=" * 60)


# Create a copy
ml_data = df.copy()


# ------------------------------------------------------------
# Convert categorical columns into numbers
# ------------------------------------------------------------

encoder = LabelEncoder()

ml_data["Country"] = encoder.fit_transform(
    ml_data["Country"]
)

ml_data["Organization"] = encoder.fit_transform(
    ml_data["Organization"]
)

ml_data["Mission_Type"] = encoder.fit_transform(
    ml_data["Mission_Type"]
)


# Encode Outcome
ml_data["Outcome"] = encoder.fit_transform(
    ml_data["Outcome"]
)


# ------------------------------------------------------------
# Select Features
# ------------------------------------------------------------

X = ml_data[
    [
        "Year",
        "Country",
        "Organization",
        "Payload_kg",
        "Mission_Type"
    ]
]

y = ml_data["Outcome"]


# ============================================================
# 14. TRAIN AND TEST DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 15. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 16. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 17. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nMachine Learning Algorithm:")
print("Random Forest Classifier")

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 19. FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)
print("                 FINAL RESULTS")
print("=" * 60)

print("Total Missions       :", total_missions)
print("Successful Missions  :", successful_missions)
print("Failed Missions      :", failed_missions)
print(
    "Overall Success Rate :",
    round(success_rate, 2),
    "%"
)
print(
    "Most Active Country  :",
    most_active_country
)
print(
    "Most Active Organization:",
    most_active_org
)
print(
    "ML Model Accuracy    :",
    round(accuracy * 100, 2),
    "%"
)

print("\n")
print("=" * 60)
print("             PROJECT COMPLETED!")
print("=" * 60)
