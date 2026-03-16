CREATE OR REPLACE TABLE mart.customer_features_daily AS
WITH dates AS (
  SELECT d::DATE AS feature_date
  FROM generate_series(CURRENT_DATE - INTERVAL 29 DAY, CURRENT_DATE, INTERVAL 1 DAY) t(d)
),
base AS (
  SELECT
    c.customerID,
    d.feature_date,

    -- static snapshot features
    c.SeniorCitizen,
    c.tenure,
    c.Contract,
    c.PaymentMethod,
    c.PaperlessBilling,
    c.InternetService,
    c.TechSupport,
    c.MonthlyCharges,
    c.TotalCharges,

    -- label (baseline uses snapshot label)
    c.churn_label

  FROM stg.customers c
  CROSS JOIN dates d
)
SELECT
  customerID,
  feature_date,

  SeniorCitizen,
  tenure,
  Contract,
  PaymentMethod,
  PaperlessBilling,
  InternetService,
  TechSupport,
  MonthlyCharges,
  TotalCharges,

  -- simple daily signals (deterministic so reruns are stable)
  (MonthlyCharges / 10.0) + (tenure / 24.0) AS usage_proxy,
  CASE
    WHEN Contract = 'Month-to-month' THEN 1 ELSE 0
  END AS is_month_to_month,
  CASE
    WHEN TechSupport = 'Yes' THEN 1 ELSE 0
  END AS has_tech_support,
  EXTRACT(DOW FROM feature_date) AS day_of_week,

  churn_label
FROM base;