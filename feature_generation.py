import pandas as pd
from datetime import date, timedelta
import numpy as np

startdate = date(2016, 1 ,1)
total_datelist = [startdate+timedelta(days=i) for i in xrange(3,21)]
train_datelist = [startdate+timedelta(days=i) for i in xrange(3,15)]
validation1_datelist = [startdate+timedelta(days=i) for i in xrange(15,20,2)]
validation2_datelist = [startdate+timedelta(days=i) for i in xrange(16,21,2)]
test_datelist = [startdate+timedelta(days=i) for i in xrange(21,30,2)]
region_table = pd.read_table("season_1/training_data/cluster_map/cluster_map", index_col=0, names=['hash', 'id'])

def customer_level(a):
    if a==0: return 0;
    elif a<=2: return 1;
    elif a<=5: return 2;
    elif a<=10: return 3;
    elif a<=50: return 4;
    else: return 5;
    
def gap_level(a):
    if a==0: return 0;
    elif a<=2: return 1;
    elif a<=5: return 2;
    elif a<=20: return 3;
    else: return 4;

def weather_feature_generation(path, datelist):
    """generated weather feature is a numpy 2d array of length 144 x len(datelist).
    Each feature is 9 elements long
    """
    # aggregration first
    weatherlist = []
    for d in datelist:
        weather = pd.read_table(path+"/weather_data/weather_data_"+d.isoformat(), 
                                names=['time','weather','temperature','pm'], parse_dates=[0])
        time_slot = np.zeros((weather.shape[0],),np.int)
        jour = np.zeros((weather.shape[0],), date)
        for i, t in enumerate(weather['time']):
            time_slot[i] = t.time().hour*6+t.time().minute/10
            jour[i] = t.date()
        weather['slot'] = time_slot
        weather['jour'] = jour
        weatherlist.append(weather)
    grouped_weatherlist = pd.concat(weatherlist).groupby(['jour', 'slot'])
    # generation second
    weather_feature = np.zeros((len(datelist)*144, 9), dtype = np.float)
    for i, dt in enumerate(datelist):
        for j in xrange(144):
            try:
                rec = grouped_weatherlist.get_group((dt, j))
                for k in xrange(rec.shape[0]):
                    weather_feature[i*144+j, rec.iloc[k,1]-1] += 1.0/rec.shape[0]
            except KeyError:
                continue;
    # post processing of weather feature, using linear interplotation
    b = -1
    for e in xrange(weather_feature.shape[0]+1):
        if e==weather_feature.shape[0] or not np.all(weather_feature[e,:]==0.0):
            if e==weather_feature.shape[0]:
                if b!=e-1:
                    for i in xrange(b+1, e):
                        weather_feature[i, :] = weather_feature[b, :]
                break;

            if not np.all(weather_feature[e,:]==0.0) and b!=e-1:
                r = e-b+0.0
                for i in xrange(b+1, e):
                    weather_feature[i, :] = (e-i)/r*weather_feature[b, :] + (i-b)/r*weather_feature[e, :]
            elif b==-1 and b!=e-1:
                for i in xrange(b+1, e):
                    weather_feature[i, :] = weather_feature[e, :]
            b=e
    return weather_feature
    
# read order records
def refine_order_list(path, datelist):
    order_table_list = []
    for d in datelist:
        order_table = pd.read_table(path+"/order_data/order_data_"+d.isoformat(), 
                                    names=['order_id', 'driver_id', 'passenger_id', 'depart_id', 'dest_id', 'price', 'time'], 
                                    parse_dates=[6])
        time_slot = np.zeros((order_table.shape[0],),np.int)
        jour = np.zeros((order_table.shape[0],), date)
        for i, t in enumerate(order_table['time']):
            time_slot[i] = t.time().hour*6+t.time().minute/10
            jour[i] = t.date()
        order_table['time_slot']=time_slot
        order_table['jour']=jour
        order_table['depart_id'] = order_table['depart_id'].apply(lambda x: region_table.ix[x, 'id'])
        order_table_list.append(order_table)
    order_table = pd.concat(order_table_list).loc[:,['jour', 'depart_id', 'time_slot', 'order_id', 'driver_id']]# all necessary information, at least what I considered to be necessary.
    f = order_table.groupby(by=['jour', 'depart_id', 'time_slot'], as_index=False).count()
    return f
    
def training_data_generation(order_table, whole_grouped_order_table, weather_feature):
    # flst is the final feature
    flst = []
    # generate time feature
    for i in xrange(order_table.shape[0]):
        x = [(order_table.ix[i, 'time_slot'], 1), (144+order_table.ix[i, 'time_slot']/6, 1), (168+order_table.ix[i, 'jour'].weekday()/5, 1)]
        flst.append(x)
    # generate region feature
    for i in xrange(order_table.shape[0]):
        flst[i].append((170+order_table.ix[i, 'depart_id']-1, 1))
    # generate customer feature
    dd = timedelta(days=7)
    for i in xrange(order_table.shape[0]):
        bd = order_table.ix[i, 'jour']-dd
        try:
            rec = whole_grouped_order_table.get_group((bd, order_table.ix[i, 'depart_id'], order_table.ix[i, 'time_slot']))
            gp = rec.iloc[0, 3]-rec.iloc[0, 4]
            demand = rec.iloc[0, 3]
            flst[i].append((236+6*gap_level(gp)+customer_level(demand), 1))
        except KeyError:
            continue;
    # weather feature
    for i in xrange(order_table.shape[0]):
        wf = weather_feature[order_table.ix[i, 'time_slot']+144*(order_table.ix[i, 'jour'].day-1), :]
        for idx in np.nonzero(wf)[0]:
            flst[i].append((266+idx, wf[idx]))
    return flst
    
def test_data_generation(filename, whole_grouped_order_table, weather_feature):
    flst = []
    dd = timedelta(days=7)
    with open(filename, 'r') as fr:
        while True:
            r = fr.readline().strip()
            if not r: break;
            
            r = [int(itm) for itm in r.split('-')]
            x = [(r[3]-1, 1), (144+(r[3]-1)/6, 1), (168+date(r[0], r[1], r[2]).weekday()/5, 1)]
            
            for t in xrange(66):
                xc = x[:]
                xc.append((170+t, 1))
                
                bd = date(r[0], r[1], r[2])-dd
                try:
                    rec = whole_grouped_order_table.get_group((bd, t+1, r[3]-1))
                    gp = rec.iloc[0, 3]-rec.iloc[0, 4]
                    demand = rec.iloc[0, 3]
                    xc.append((236+6*gap_level(gp)+customer_level(demand), 1))
                except KeyError:
                    pass;
                    
                wf = weather_feature[r[3]-1+144*((r[2]-22)/2), :]
                for idx in np.nonzero(wf)[0]:
                    xc.append((266+idx, wf[idx]))
                    
                flst.append(xc)
    return flst
    
def run():
    weather_feature = weather_feature_generation("season_1/training_data", [startdate+timedelta(days=i) for i in xrange(0,21)])
    # get region table
    total_order = refine_order_list("season_1/training_data", total_datelist)
    total_grouped_order = total_order.groupby(['jour', 'depart_id', 'time_slot'])
    flst = training_data_generation(total_order, total_grouped_order, weather_feature)
    
    rst = total_order.jour.isin(train_datelist)
    train_feature = [flst[i] for i in rst[rst].index]  
    with open("training_data", 'w') as fw:
        for f, i in zip(train_feature, rst[rst].index):
            if total_order.ix[i, 'order_id']-total_order.ix[i, 'driver_id']==0:
                continue;
            s = '{0} {1} {2} {3} '.format(total_order.ix[i, 'order_id']-total_order.ix[i, 'driver_id'],
            total_order.ix[i, 'jour'].day, 
            total_order.ix[i, 'time_slot'], 
            total_order.ix[i, 'depart_id'])
            for idx, val in f:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
    
    rst = total_order.jour.isin(validation1_datelist)
    valid1_feature = [flst[i] for i in rst[rst].index]
    with open("validation1_data", 'w') as fw:
        for f, i in zip(valid1_feature, rst[rst].index):
            s = '{0} {1} {2} {3} '.format(total_order.ix[i, 'order_id']-total_order.ix[i, 'driver_id'],
            total_order.ix[i, 'jour'].day, 
            total_order.ix[i, 'time_slot'], 
            total_order.ix[i, 'depart_id'])
            for idx, val in f:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
    
    rst = total_order.jour.isin(validation2_datelist)
    valid2_feature = [flst[i] for i in rst[rst].index]
    with open("validation2_data", 'w') as fw:
        for f, i in zip(valid2_feature, rst[rst].index):
            s = '{0} {1} {2} {3} '.format(total_order.ix[i, 'order_id']-total_order.ix[i, 'driver_id'],
            total_order.ix[i, 'jour'].day, 
            total_order.ix[i, 'time_slot'], 
            total_order.ix[i, 'depart_id'])
            for idx, val in f:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
            
def run_test():
    weather_feature = weather_feature_generation("season_1/training_data", total_datelist)
    # get region table
    train_order = refine_order_list("season_1/training_data", train_datelist+validation_datelist)
    total_grouped_train_order = train_order.groupby(['jour', 'depart_id', 'time_slot'])
    flst = training_data_generation(train_order, total_grouped_train_order, weather_feature)
        
    with open("training_data_total", 'w') as fw:
        for i, features in enumerate(flst):
            s = '{0} {1} {2} {3} '.format(train_order.ix[i, 'order_id']-train_order.ix[i, 'driver_id'],
            train_order.ix[i, 'jour'].day, 
            train_order.ix[i, 'time_slot'], 
            train_order.ix[i, 'depart_id'])
            for idx, val in features:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)

    weather_feature = weather_feature_generation("season_1/test_set_1/", test_datelist)
    flst = test_data_generation("season_1/test_set_1/read_me_1.txt", total_grouped_train_order, weather_feature)
    
    with open("test_data", 'w') as fw:
        for features in flst:
            s = ''
            for idx, val in features:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
    
            
if __name__=='__main__':
    run()