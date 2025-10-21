# Exploratory Data Analysis Report

*Generated: 2025-10-20 21:00:32*

## Dataset Overview
- **Rows:** 208
- **Columns:** 5
- **Memory:** 0.02 MB
- **Numeric columns:** 4
- **Categorical columns:** 1

## Data Quality
- **Missing values:** 0.00%
- **Complete rows:** 100.0%
- **Duplicate rows:** 0.00%

## Key Insights

## Feature Engineering Suggestions

### 1. Create categorical bins for media_activity
**Type:** binning
**Reasoning:** media_activity has many unique values, binning may capture non-linear patterns

### 2. Create categorical bins for competitor_actions
**Type:** binning
**Reasoning:** competitor_actions has many unique values, binning may capture non-linear patterns

### 3. Create categorical bins for sales
**Type:** binning
**Reasoning:** sales has many unique values, binning may capture non-linear patterns

## Generated Files
- `statistics/`: All statistical analyses
- `plots/`: All visualizations
- `insights.json`: Extracted insights
- `feature_suggestions.json`: Feature engineering recommendations