# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the data
diabetes_df = pd.read_csv("datasetForTask_f.csv")
diabetes_df

# Histogram of diabetes prevalence by country
plt.hist(diabetes_df['Age-standardised diabetes prevalence'], bins=20)
plt.xlabel('Diabetes prevalence')
plt.ylabel('Number of countries')
plt.show()

# Scatterplot of diabetes prevalence by year
plt.scatter(diabetes_df['Year'], diabetes_df['Age-standardised diabetes prevalence'])
plt.xlabel('Year')
plt.ylabel('Diabetes prevalence')
plt.show()

# Boxplot of diabetes prevalence by region
plt.boxplot([diabetes_df.loc[diabetes_df['ISO'] == iso]['Age-standardised diabetes prevalence'] for iso in np.unique(diabetes_df['ISO'])])
plt.xticks(range(1, len(np.unique(diabetes_df['ISO'])) + 1), np.unique(diabetes_df['ISO']), rotation=90)
plt.xlabel('Region')
plt.ylabel('Diabetes prevalence')
plt.show()

# Heatmap of diabetes prevalence by age and sex
heatmap_data = diabetes_df.pivot_table(index='Year', columns='Sex', values='Age-standardised diabetes prevalence', aggfunc=np.mean)
plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.xlabel('Sex')
plt.ylabel('Year')
plt.show()

# Create a bar chart of the top countries with the highest diabetes prevalence
top10_df = diabetes_df.sort_values("Age-standardised diabetes prevalence", ascending=False).head(10)
sns.barplot(data=top10_df, x="Country/Region/World", y="Age-standardised diabetes prevalence")
plt.xlabel('Country')
plt.show()

# Create a histogram of the uncertainty intervals
sns.histplot(data=diabetes_df, x="Upper 95% uncertainty interval")
plt.show()

# Create a violin plot of diabetes prevalence by sex
sns.violinplot(data=diabetes_df, x="Sex", y="Age-standardised diabetes prevalence")
plt.show()

# Create a box plot of diabetes prevalence by age group
sns.boxplot(data=diabetes_df, x="Sex", y="Age-standardised diabetes prevalence")
plt.show()

