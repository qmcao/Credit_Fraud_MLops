


Question: Do we need to encode binary variable?

Treat truly numeric features with a numeric pipeline (median imputation, robust scaling, and optional numeric feature selection).
Treat multi-category nominal features (5–7 unique categories) with a categorical pipeline (most frequent imputation, one-hot encoding, and optional feature selection by Chi square
  or mutual information).
Treat binary features (already 0/1) either as-is (pass through) or with a minimal pipeline (e.g., simple imputation if missing). Typically, you won’t scale them or run one-hot encoding on them, since they’re already 0/1.