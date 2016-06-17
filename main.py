from evaluation import *
import xgboost as xgb


def run():
    X, Y, slot, dist = getdata("training_data_total", "lr_train_total")
    Xt, recs = gettestdata("test_data", "season_1/test_set_2/read_me_2.txt", "lr_test_rst")
    weight = np.zeros(Y.shape)
    for i, y in enumerate(Y):
        if y==0:
            weight[i]=0
        else:
            weight[i]=1.0/y
    dtrain = xgb.DMatrix(X, label=Y, weight=weight)
    dtest = xgb.DMatrix(Xt)

    gbdr = xgb.train({"max_depth":6,"objective":"reg:linear", "eval_metric":"mae", "min_child_weight":3, 'alpha':5, 'lambda':1,'gamma':0, 'subsample':0.85}, dtrain, num_boost_round=3)
    #lr = xgb.train({"booster":"gblinear", 'lambda':5,'alpha':5}, dtrain)
    
    Yt_pred = gbdr.predict(dtest)
    Yt_pred[Yt_pred<0]=0
    
    with open("result.csv", "w") as fw:
        for i, (r, s) in enumerate(recs):
            fw.write("{0},{1},{2}\n".format(r, s, Yt_pred[i]))
            
if __name__=='__main__':
    run()