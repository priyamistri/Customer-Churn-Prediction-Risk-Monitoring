-- Stores top SHAP feature drivers per customer per score date
CREATE TABLE IF NOT EXISTS ml.churn_explanations (
  score_date DATE,
  customerID VARCHAR,
  model_name VARCHAR,
  model_version VARCHAR,

  base_value DOUBLE,
  churn_probability DOUBLE,

  -- JSON text for easy storage in DuckDB
  top_features_json VARCHAR,
  all_features_json VARCHAR
);