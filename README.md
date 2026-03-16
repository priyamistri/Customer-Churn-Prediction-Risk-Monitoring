# Customer Churn ETL Workflow

## Objective
Build a customer churn project with a complete ETL workflow for a portfolio project.

## Data Extraction
- Loaded customer churn data from a CSV file
- Used Python to read the raw data

## Data Cleaning
- Checked and handled missing values
- Fixed data types where needed
- Removed duplicate records if present
- Prepared the data for the next steps

## Data Transformation
- Selected useful columns for churn analysis
- Converted raw data into a cleaner format
- Organized the data so it could be loaded and used easily

## ETL Workflow
- Built the ETL flow using Python
- Used Prefect to organize and run the workflow
- Created a main flow file for the churn pipeline
- Set up the project so the flow can be run from the terminal

## Tools Used
- Python
- Pandas
- Prefect
- VS Code
- Git and GitHub

## Project Structure
```text
churn-etl-dashboard/
│
├── flows/
├── run_flow.py
├── prefect.yaml
├── requirements.txt
└── README.md
