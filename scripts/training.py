# We utilize the best hyperparameters which we found with optuna to train our model

gbm = lgb.LGBMRanker(
    n_estimators=10000,
    num_leaves=75,
    learning_rate=0.09183351606138195,
    reg_lambda=5.910358251333725,
    max_depth=6,
    min_data_in_leaf=4700,
    max_bin=438,

)

gbm.fit(
    split["X_train"],
    y_train,
    group=train_group,
    eval_group=[vali_group],
    eval_set=[(split["X_vali"],y_val)],
    early_stopping_rounds=150
)
