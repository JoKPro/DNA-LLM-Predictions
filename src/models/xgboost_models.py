

train_X, train_y = next(iter(embedding_dataset.train_dataloader()))
val_X, val_y = next(iter(embedding_dataset.val_dataloader()))
test_X, test_y = next(iter(embedding_dataset.test_dataloader()))

dtrain = xgb.DMatrix(train_X, train_y)
dval = xgb.DMatrix(val_X, val_y)
dtest = xgb.DMatrix(test_X, test_y)

evals = [(dtrain, "train"), (dval, "eval")]

params = {
    "objective": "reg:logistic", 
    "eval_metric": "auc",
    "tree_method": "gpu_hist"
}
n = 100

model = xgb.train(
    params=params, 
    dtrain=dtrain, 
    evals=evals,
    num_round=n, 
    early_stopping_rounds=10
)

train_preds = (model.predict(dtrain, iteration_range=(0, model.best_iteration + 1)), train_y, "train")
val_preds = (model.predict(dval, iteration_range=(0, model.best_iteration + 1)), val_y, "val")


class XgbModel: 
    
    def __init__(self, 
                 param, 
                 num_round, 
                 early_stopping_rounds=None, 
                 cv=False, 
                 nfold=5, 
                 
                 
            ): 
        self.save_hyperparameters()

    
    def setup_data(self, data: L.LightningDataModule)
        train_X, train_y = next(iter(data.train_dataloader()))
        val_X, val_y = next(iter(self.hparams.data.val_dataloader()))
        test_X, test_y = next(iter(self.hparams.data.test_dataloader()))
    
    def train(self): 
        
        if self.hparams.cv: 
            model = xgb.cv(
                params=self.hparams.params, 
                dtrain=self.dtrain, 
                evals=self.evals, 
                num_boost_round=self.hparams.num_round, 
                early_stopping_rounds=self.hparams.early_stopping_rounds
            )
        else: 
            model = xgb.train(
                params=self.hparams.params, 
                dtrain=self.dtrain, 
                evals=self.evals, 
                num_round=self.hparams.num_round, 
                early_stopping_rounds=self.hparams.early_stopping_rounds
            )
        return model
        
    
