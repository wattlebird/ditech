import numpy as np
from sklearn.linear_model import Ridge
from scipy.sparse import coo_matrix
from feature_generation import gap_level

sz=np.array([24, 2, 66, 21, 21, 49, 7])
combpattern = np.array([
    [0,1,1,1,1,1,1],
    [0,0,1,1,1,1,1],
    [0,0,0,1,1,1,1],
    [0,0,0,0,1,0,1],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1],
    [0,0,0,0,0,0,0]
])

offset = np.zeros(sz.shape, np.int)
offset[0]=sz[0]
for i in xrange(1, offset.shape[0]):
    offset[i] = sz[i]+offset[i-1]

ext = np.zeros(combpattern.shape, np.int)
l = offset[-1]
for i in xrange(ext.shape[0]):
    for j in xrange(1, ext.shape[1]):
        if combpattern[i, j]!=0:
            ext[i, j]=l
            l += sz[i]*sz[j]
        
def extend_2dfeature(row, data):
    exrow = row[:]
    exdata = data[:]
    for i, r in enumerate(row):
        o = np.sum(offset<=r)
        for j in xrange(i+1, len(row)):
            if row[j]>=offset[o]:
                q = np.sum(offset<=row[j])
                off = ext[o, q]
                if off==0: continue
                if o>0:
                    exrow.append(off+(r-offset[o-1])*sz[q]+row[j]-offset[q-1])
                else:
                    exrow.append(off+r*sz[q]+row[j]-offset[q-1])
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
    
def getdata(filename, lrresult, extent2d=True):
    y_true = []
    col = []
    data = []

    date = []
    slot = []
    dist = []

    with open(filename, 'r') as fr, open(lrresult, 'r') as fx:
        while True:
            event = fr.readline().strip()
            if not event: break;
            lrpred = float(fx.readline().strip())
            
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
            vidx.append(offset[-2]+gap_level(lrpred))
            vdata.append(1)
            col.append(vidx)
            data.append(vdata)
            
    for i, (c, d) in enumerate(zip(col, data)):
        if extent2d:
            c, d = extend_2dfeature(c, d)
        col[i]=c
        data[i]=d
        
    row = []
    for i in xrange(len(col)):
        row.append([i]*len(col[i]))
        
    if extent2d:
        X = coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(len(y_true), l))
        X = X.tocsr()
    else:
        X = coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(len(y_true), offset[-1]))
    Y = np.asarray(y_true)
    
    return (X, Y, slot, dist)

def gettestdata(testdata, testdescription, lrresult, extent2d=True):
    record = []
    with open(testdescription, 'r') as fr:
        while True:
            r = fr.readline().strip()
            if not r: break;
            
            for i in xrange(1,67):
                record.append((i, r))
    
    col = []
    data = []
    with open(testdata, 'r') as fr, open(lrresult) as fx:
        while True:
            r = fr.readline().strip()
            if not r: break;
            r = r.split(" ")
            lrpred = float(fx.readline().strip())
            
            vidx = []
            vdata = []
            for x in r:
                idx, val = x.split(":")
                vidx.append(int(idx))
                vdata.append(float(val))
            vidx.append(offset[-2]+gap_level(lrpred))
            vdata.append(1)
            col.append(vidx)
            data.append(vdata)
            
    for i, (c, d) in enumerate(zip(col, data)):
        if extent2d:
            c, d = extend_2dfeature(c, d)
        col[i]=c
        data[i]=d
        
    row = []
    for i in xrange(len(col)):
        row.append([i]*len(col[i]))
        
    if extent2d:
        X = coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(len(record), l))
        X = X.tocsr()
    else:
        X = coo_matrix((np.hstack(data), (np.hstack(row), np.hstack(col))), shape=(len(record), offset[-1]))
    
    return (X, record)