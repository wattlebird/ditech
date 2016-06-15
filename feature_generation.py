import pandas as pd
from datetime import date, timedelta
import numpy as np

startdate = date(2016, 1 ,1)
total_datelist = [startdate+timedelta(days=i) for i in xrange(3,21)]
train_datelist = [startdate+timedelta(days=i-1) for i in [5,6,7,8,10,10,11,12,13,14,15,16,16,18,20]]
validation_datelist = [startdate+timedelta(days=i-1) for i in [4,9,17,19,21]]
test_datelist = [startdate+timedelta(days=i) for i in xrange(22,31,2)]
total_training_datelist = [startdate+timedelta(days=i-1) for i in [4,5,6,7,8,9,10,10,11,12,13,14,15,16,16,17,18,19,20,21]]
region_table = pd.read_table("season_1/training_data/cluster_map/cluster_map", index_col=0, names=['hash', 'id'])

def gap_level(a):
    if a==0: return 0;
    elif a<=1: return 1;
    elif a<=3: return 2;
    elif a<=6: return 3;
    elif a<=12: return 4;
    elif a<=30: return 5;
    else: return 6;
    
def customer_level(a):
    if a<=10: return 0;
    elif a<=20: return 1;
    else: return 2;

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
    
def traffic_generation(path, datelist):
    vtraffic = []
    for d in datelist:
        traffic_table = pd.read_table(path+"/traffic_data/traffic_data_"+d.isoformat(), names=['district', 
                'one', 'two', 'three', 'four', 'time'], parse_dates=[5])
        traffic = traffic_table[['one', 'two', 'three', 'four']].applymap(lambda x: int(x.split(':')[1]))
        traffic = traffic.div(traffic.sum(axis=1), axis=0)
        traffic['slot'] = traffic_table.time.apply(lambda x: x.hour*6+x.minute/10)
        traffic['date'] = traffic_table.time.apply(lambda x: x.day)
        traffic['district'] = traffic_table.district.apply(lambda x: region_table.ix[x, 0])
        vtraffic.append(traffic)
    grouped_traffic = pd.concat(vtraffic).groupby(['date','district','slot'])
    return grouped_traffic
    
def training_first_order(order_table):
    grouped_order = order_table.groupby('depart_id')
    delta = []
    for destrict in xrange(1, 67):
        destrict_order = grouped_order.get_group(destrict).copy()
        gap = destrict_order['order_id']-destrict_order['driver_id']
        dgap = np.zeros(gap.shape)
        for i in xrange(1, dgap.shape[0]):
            dgap[i] = gap.iloc[i]-gap.iloc[i-1]
        ddemand = np.zeros(destrict_order['order_id'].shape)
        for i in xrange(1, ddemand.shape[0]):
            ddemand[i] = destrict_order['order_id'].iloc[i] - destrict_order['order_id'].iloc[i-1]
        destrict_order['dgap']=dgap
        destrict_order['ddemand']=ddemand
        delta.append(destrict_order)
    dor = pd.concat(delta)
    return dor
    
def test_first_order(order_table, datelist):
    grouped_order = order_table.groupby(['jour', 'depart_id'])
    delta = []
    for dt in datelist:
        for destrict in xrange(1, 67):
            destrict_order = grouped_order.get_group((dt, destrict)).copy()
            gap = destrict_order['order_id']-destrict_order['driver_id']
            dgap = np.zeros(gap.shape)
            for i in xrange(1, dgap.shape[0]):
                dgap[i] = gap.iloc[i]-gap.iloc[i-1]
            ddemand = np.zeros(destrict_order['order_id'].shape)
            for i in xrange(1, ddemand.shape[0]):
                ddemand[i] = destrict_order['order_id'].iloc[i] - destrict_order['order_id'].iloc[i-1]
            destrict_order['dgap']=dgap
            destrict_order['ddemand']=ddemand
            delta.append(destrict_order)
    dor = pd.concat(delta)
    return dor

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
    f = order_table.groupby(by=['jour', 'time_slot', 'depart_id'], as_index=False).count()
    f['gap'] = f['order_id']-f['driver_id']
    return f
    
def training_data_generation(demand, gap, datelist):
    flst = [None]*(len(datelist)*144*66)
    days = np.zeros(len(datelist)*144*66, dtype=np.int)
    y = np.zeros(len(datelist)*144*66, dtype=np.int)
    slots = np.zeros(len(datelist)*144*66, dtype=np.int)
    districts = np.zeros(len(datelist)*144*66, dtype=np.int)
    for i, d in enumerate(datelist):
        curd = d.day-4 # 4 as offset
        pred = curd-7
        weekd = d.weekday()/5
        
        days[i*144*66:(i+1)*144*66] = d.day
        for t in xrange(144):
            hour = t/6;
            offset = curd*144*66+t*66
            slots[i*144*66+t*66:i*144*66+t*66+66] = t
            districts[i*144*66+t*66:i*144*66+t*66+66] = np.arange(1, 67)
            
            #generate spatial feature first
            #sf = []
            #if offset>=66:
            #    for r in xrange(66):
            #        sf.append((183+r*7+gap_level(gap[offset-66+r]), 1))
            
            for r in xrange(66):
                y[i*144*66+t*66+r] = gap[offset+r]
                flst[i*144*66+t*66+r] = [(hour, 1), (24+weekd, 1), (26+r, 1)]
                if pred>=0:
                    flst[i*144*66+t*66+r].append((92+3*gap_level(gap[pred*144*66+t*66+r])+
                                                customer_level(demand[pred*144*66+t*66+r]), 1))
                if offset>=66:
                    flst[i*144*66+t*66+r].append((113+3*gap_level(gap[offset-66+r])+
                                                customer_level(demand[offset-66+r]), 1))
                                                
                if offset>=132:
                    flst[i*144*66+t*66+r].append((134+7*gap_level(gap[offset-132+r])+
                                                gap_level(gap[offset-66+r]), 1))
                                                
                                                
                #flst[i*144*66+t*66+r].extend(sf)
    return flst, y, days, slots, districts
                
def test_data_generation(filename, tdemand, tgap, demand, gap):
    flst = []
    with open(filename, 'r') as fr:
        while True:
            r = fr.readline().strip()
            if not r: break;
            
            r = [int(itm) for itm in r.split('-')]
            weekd = date(r[0], r[1], r[2]).weekday()/5
            offset = (r[2]-23)/2*144*66 + (r[3]-1)*66
            #sf = []
            #for d in xrange(66):
            #    sf.append((183+d*7+gap_level(gap[offset-66+d]), 1))
                
            for d in xrange(66):
                flst.append([(r[3]/6, 1), (24+weekd, 1), (26+d, 1)])
                if r[2]-7<=21:
                    flst[-1].append((92+3*gap_level(tgap[(r[2]-7-4)*144*66+(r[3]-1)*66+d])+
                                 customer_level(tdemand[(r[2]-7-4)*144*66+(r[3]-1)*66+d]), 1))
                flst[-1].append((113+3*gap_level(gap[offset+d-66])+customer_level(demand[offset+d-66]), 1))
                flst[-1].append((134+7*gap_level(gap[offset+d-66-66])+gap_level(gap[offset+d-66]), 1))
                #flst[-1].extend(sf)
                
    return flst

            
    
def run():
    #weather_feature = weather_feature_generation("season_1/training_data", [startdate+timedelta(days=i) for i in xrange(0,21)])
    # get region table
    total_order = refine_order_list("season_1/training_data", total_datelist)
    demand = np.zeros(len(total_datelist)*144*66, dtype=np.int)
    gap = np.zeros(len(total_datelist)*144*66, dtype=np.int)
    for idx, row in total_order.iterrows():
        demand[144*66*(row['jour'].day-4)+66*row['time_slot']+row['depart_id']-1]=row['order_id']
        gap[144*66*(row['jour'].day-4)+66*row['time_slot']+row['depart_id']-1]=row['gap']
    flst, y, days, slots, destricts = training_data_generation(demand, gap, train_datelist)
    
    with open("training_data", 'w') as fw:
        for f, gp, day, slot, dest in zip(flst, y, days, slots, destricts):
            if gp==0: continue;
            s = '{0} {1} {2} {3} '.format(gp, day, slot, dest)
            for idx, val in f:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
    
    flst, y, days, slots, destricts = training_data_generation(demand, gap, validation_datelist)
    with open("validation_data", 'w') as fw:
        for f, gp, day, slot, dest in zip(flst, y, days, slots, destricts):
            if gp==0: continue;
            s = '{0} {1} {2} {3} '.format(gp, day, slot, dest)
            for idx, val in f:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
    
            
def run_test():
    #weather_feature = weather_feature_generation("season_1/training_data", [startdate+timedelta(days=i) for i in xrange(0,21)])
    # get region table
    total_order = refine_order_list("season_1/training_data", total_datelist)
    demand = np.zeros(len(total_datelist)*144*66, dtype=np.int)
    gap = np.zeros(len(total_datelist)*144*66, dtype=np.int)
    for idx, row in total_order.iterrows():
        demand[144*66*(row['jour'].day-4)+66*row['time_slot']+row['depart_id']-1]=row['order_id']
        gap[144*66*(row['jour'].day-4)+66*row['time_slot']+row['depart_id']-1]=row['gap']
    flst, y, days, slots, destricts = training_data_generation(demand, gap, total_training_datelist)
        
    with open("training_data_total", 'w') as fw:
        for f, gp, day, slot, dest in zip(flst, y, days, slots, destricts):
            if gp==0: continue;
            s = '{0} {1} {2} {3} '.format(gp, day, slot, dest)
            for idx, val in f:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)

    test_order = refine_order_list("season_1/test_set_2", test_datelist)
    tdemand = np.zeros(len(test_datelist)*144*66, dtype=np.int)
    tgap = np.zeros(len(test_datelist)*144*66, dtype=np.int)
    for idx, row in test_order.iterrows():
        tdemand[144*66*(row['jour'].day-23)/2+66*row['time_slot']+row['depart_id']-1]=row['order_id']
        tgap[144*66*(row['jour'].day-23)/2+66*row['time_slot']+row['depart_id']-1]=row['gap']
    flst = test_data_generation("season_1/test_set_2/read_me_2.txt", demand, gap, tdemand, tgap)
    
    with open("test_data", 'w') as fw:
        for features in flst:
            s = ''
            for idx, val in features:
                if val!=0: s+='{0}:{1} '.format(idx, val)
            s+='\n'
            fw.write(s)
    
            
if __name__=='__main__':
    run_test()