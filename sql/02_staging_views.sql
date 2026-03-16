CREATE OR REPLACE VIEW stg.customers AS
SELECT
  customerID,
  gender,
  SeniorCitizen,
  Partner,
  Dependents,
  tenure,
  PhoneService,
  MultipleLines,
  InternetService,
  OnlineSecurity,
  OnlineBackup,
  DeviceProtection,
  TechSupport,
  StreamingTV,
  StreamingMovies,
  Contract,
  PaperlessBilling,
  PaymentMethod,
  MonthlyCharges,
  TRY_CAST(NULLIF(TotalCharges, '') AS DOUBLE) AS TotalCharges,
  CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS churn_label
FROM raw.telco_customers_snapshot;