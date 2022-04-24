import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

test_pred = gbm.predict(split['X_test']) # Find model predictions
true_relevance = y_test.drop("predicted_ranking", axis=1) # Drop column to fit necessary dimensions
relevance_score = y_test.sort_values("predicted_ranking", ascending=False)

# Use variables to calculate nDCG score

print(
        "nDCG score: ",
        ndcg_score(
            [true_relevance.to_numpy().reshape(241520,)], [relevance_score["relevance_score"].to_numpy()]
        ),
    )
