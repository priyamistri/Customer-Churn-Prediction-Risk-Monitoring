CREATE TABLE IF NOT EXISTS ml.model_monitoring_daily (
  score_date DATE,
  model_name VARCHAR,
  model_version VARCHAR,

  n_scored INTEGER,
  avg_proba DOUBLE,
  std_proba DOUBLE,
  p95_proba DOUBLE,

  psi_vs_prev DOUBLE,
  missing_totalcharges_rate DOUBLE,
  missing_monthlycharges_rate DOUBLE,

  created_at TIMESTAMP
);