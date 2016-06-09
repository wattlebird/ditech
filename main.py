from evaluation import *
from sklearn.linear_model import Ridge

def run():
    X, Y, slot, dist = getdata("training_data_total")
    weight = np.zeros(Y.shape)
    for i, y in enumerate(Y):
        if y==0:
            weight[i]=1
        else:
            weight[i]=1/y

    lr = Ridge(alpha=10)
    lr.fit(X, Y, weight)
    
    Xt, recs = gettestdata("test_data", "season_1/test_set_1/read_me_1.txt")
    Yt_pred = lr.predict(Xt)
    Yt_pred[Yt_pred<0]=0
    
    with open("result.csv", "w") as fw:
        for i, (r, s) in enumerate(recs):
            fw.write("{0},{1},{2}\n".format(r, s, Yt[i]))
            
if __name__=='__main__':
    run()