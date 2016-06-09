from evaluation import *
from sklearn.linear_model import Ridge
from scipy.sparse import coo_matrix

def getdata(filename):
    """
    return the model, Y, time slot, and distriction
    """
    y_true = []
    col = []
    data = []

    date = []
    slot = []
    dist = []

    with open(filename, 'r') as fr:
        while True:
            event = fr.readline().strip()
            if not event: break;
            
            event = event.split(' ')
            y_true.append(int(event[0]))
            date.append(int(event[1]))
            slot.append(int(event[2]))
            dist.append(int(event[3]))
            
            vidx = []
            vdata = []
            for itm in event[4:]:
                idx, val = itm.split(':')
                vidx.append(int(idx))
                vdata.append(float(val))
            col.append(vidx)
            data.append(vdata)
            
    for i, (c, d) in enumerate(zip(col, data)):
        c, d = extend_2dfeature(c, d)
        col[i]=c
        data[i]=d
        
    row = []
    for i in xrange(len(col)):
        row.append([i]*len(col[i]))
        
    X = coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(len(y_true), ext[-1]))
    X = X.tocsr()
    Y = np.asarray(y_true)
    
    weight = np.zeros(Y.shape)
    for i, y in enumerate(Y):
        if y==0:
            weight[i]=1
        else:
            weight[i]=1/y
            
    lr = Ridge(alpha=10)
    lr.fit(X, Y, weight)
    
    return (lr, Y, slot, dist)