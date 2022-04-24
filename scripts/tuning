!pip install optuna
from typing_extensions import ParamSpec
import optuna.integration.lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback

def objective(trial):
  
  param = {
    "n_estimators": trial.suggest_categorical("n_estimators",[10000]),
    "num_leaves": trial.suggest_int("num_leaves",5,125),
    "learning_rate": trial.suggest_float("learning_rate",0.0001,0.2),
    "reg_lambda": trial.suggest_float("reg_lambda",0,10),
    "max_depth": trial.suggest_int("max_depth",3,12),
    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf",200,10000,step=100),
    "max_bin": trial.suggest_int("max_bin",300,550),
}

# We use LightGBM to train our model as it is efficient and supports learn to rank scenarios
# We use optuna to find the best hyperparameters for our model as it supports LightGBM
# Some of the more important paramaters within our model are num_leaves(controls model complexity),
# min_data_in_leaf(prevents over-fitting) and max_depth(limit tree depth explicitly)

  gbm = lgb.LGBMRanker(
      num_leaves=param["num_leaves"],
      learning_rate=param["learning_rate"],
      reg_lambda=param["reg_lambda"],
      max_depth = param["max_depth"],
      min_data_in_leaf = param["min_data_in_leaf"],
      max_bin = param["max_bin"],
      n_estimators = param["n_estimators"],
  )
  gbm.fit(
      split["X_train"],
      y_train,
      group=train_group,
      eval_group=[vali_group],
      eval_set=[(split["X_vali"], y_val)],
      early_stopping_rounds=25,
       callbacks= [LightGBMPruningCallback(trial, "ndcg@1")],
  )
  return gbm.best_score_["valid_0"]["ndcg@1"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
best_params=study.best_trial.params

print("Number of finished trials:", len(study.trials))
print("Best trial:", best_params)
