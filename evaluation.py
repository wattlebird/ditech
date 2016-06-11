import numpy as np
from sklearn.linear_model import Ridge
from scipy.sparse import coo_matrix

offset=np.array([170, 236, 296, 359])#, 335, 339])
ext = [offset[-1]]
for i in xrange(len(offset)-1):
    if i==0:
        ext.append(ext[i]+offset[i]*(offset[-1]-offset[i]))
    else:
        ext.append(ext[i]+(offset[i]-offset[i-1])*(offset[-1]-offset[i]))
        
        
        
def extend_2dfeature(row, data):
    exrow = row[:]
    exdata = data[:]
    for i, r in enumerate(row):
        o = np.sum(offset<=r) # for the following, feature idx above than offset[o] should be activated
        for j in xrange(i+1, len(row)):
            if row[j]>=offset[o]:
                if o>0:
                    exrow.append((r-offset[o-1])*(offset[-1]-offset[o])+row[j]-offset[o]+ext[o])
                else:
                    exrow.append(r*(offset[-1]-offset[o])+row[j]-offset[o]+ext[o])
                exdata.append(data[i]*data[j])
    return (exrow, exdata)
        
def mape(y_true, y_pred, slot, dist):
    vs = np.zeros(66)
    dct = 0
    loss = np.zeros((144, 66))
    for i, (s, d) in enumerate(zip(slot, dist)):
        d -=1
        if y_true[i]!=0:
            vs[d]+=1
            loss[s, d] += np.abs(1-y_pred[i]/y_true[i])
    l = np.sum(loss, axis=0)
    rtn=0
    for i in xrange(l.shape[0]):
        if vs[i]!=0:
            dct+=1
            rtn+=l[i]/vs[i]
    return rtn/dct
    
def getdata(filename):
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
    
    return (X, Y, slot, dist)

def gettestdata(testdata, testdescription):
    record = []
    with open(testdescription, 'r') as fr:
        while True:
            r = fr.readline().strip()
            if not r: break;
            
            for i in xrange(1,67):
                record.append((i, r))
    
    col = []
    data = []
    with open(testdata, 'r') as fr:
        while True:
            r = fr.readline().strip()
            if not r: break;
            r = r.split(" ")
            
            vidx = []
            vdata = []
            for x in r:
                idx, val = x.split(":")
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
        
    X = coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(len(record), ext[-1]))
    X = X.tocsr()
    
    return (X, record)