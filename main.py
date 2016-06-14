from evaluation import *
from sklearn.linear_model import Ridge
#from sklearn.ensemble import RandomForestRegressor

def run():
    X, Y, slot, dist = getdata("training_data_total")
    weight = np.zeros(Y.shape)
    for i, y in enumerate(Y):
        if y==0:
            weight[i]=0
        else:
            weight[i]=1.0/y

    lr = Ridge(alpha=10)
    lr.fit(X, Y, weight)
    #rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=20, n_jobs=-1)
    #rf.fit(X, Y, weight)
    
    Xt, recs = gettestdata("test_data", "season_1/test_set_2/read_me_2.txt")
    Yt_pred = lr.predict(Xt)
    Yt_pred[Yt_pred<0]=0
    
    with open("result.csv", "w") as fw:
        for i, (r, s) in enumerate(recs):
            fw.write("{0},{1},{2}\n".format(r, s, Yt_pred[i]))
            
if __name__=='__main__':
    run()