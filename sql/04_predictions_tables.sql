CREATE TABLE IF NOT EXISTS ml.churn_predictions (
  score_date DATE,
  customerID VARCHAR,
  churn_probability DOUBLE,
  model_name VARCHAR,
  model_version VARCHAR
);