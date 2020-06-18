# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:45:34 2020

@author: wyzhou93
"""
import pandas as pd
from datetime import date
import os
import numpy as np
from datetime import date
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import pearsonr

holidays = [pd.Timestamp(date(2020,1,1)),pd.Timestamp(date(2020,1,20)),pd.Timestamp(date(2020,2,17))]
def RM_weekends_holidays(df, holidays):
    df['date'] = df.apply(lambda z: pd.to_datetime(z.date), axis=1)
    df['day_of_week'] = df['date'].dt.day_name()
    df = df[df['day_of_week'].isin(['Wednesday', 'Thursday', 'Friday', 'Monday','Tuesday'])]
    df = df[~df['date'].isin(holidays)]
    return df


def ExtAttr_AllCounties(county_OD_EXT):
    dat_lst = county_OD_EXT['date'].unique()
    # calculate the day-to-day trip attraction of all counties 
    result = []
    for dat in dat_lst:
        temp = county_OD_EXT[county_OD_EXT['date']==dat]
        temp = temp.groupby(['CTFIPS_2'])['Volume'].agg('sum').reset_index(name='Volume_atr')
        temp['date'] = dat
        result.append(temp)
        print(dat)
    result = pd.concat(result)
    return result


def Avg_ExtAttr_AllCounties(county_OD_EXT, start_date, end_date):
    dat_lst = county_OD_EXT['date'].unique()
    result = []
    for dat in dat_lst:
        temp = county_OD_EXT[county_OD_EXT['date']==dat]
        temp = temp.groupby(['CTFIPS_2'])['Volume'].agg('sum').reset_index(name='Volume_atr')
        temp['date'] = dat
        result.append(temp)
        print(dat)
    result = pd.concat(result)
    var_name = 'VolAtr_'+str(start_date.month)+str(start_date.day)+'_'+str(end_date.month)+str(end_date.day)
    result2 = result[(result['date'] >= start_date) & (result['date'] <= end_date)]
    result2 = result2.groupby(['CTFIPS_2'])['Volume_atr'].agg('mean').reset_index(name=var_name)
    return result2


def ExtGen(county_OD_EXT, level, start_date, end_date):
    dat_lst = county_OD_EXT['date'].unique()
    result = []
    for dat in dat_lst:
        temp = county_OD_EXT[county_OD_EXT['date']==dat]
        temp = temp.groupby(['CTFIPS_1'])['Volume'].agg('sum').reset_index(name='Volume_gen')
        temp['date'] = dat
        result.append(temp)
        print(dat)
    result = pd.concat(result)    
    if level == 'NYC':
        result = result[result['CTFIPS_1'].isin(['36061', '36047', '36005', '36081', '36085'])] # nycFips = ['36061', '36047', '36005', '36081', '36085']
        result = result.groupby(['date'])['Volume_gen'].agg('sum').reset_index(name='Volume_gen_NYC')
        # calculate the weeday average given a time period
        result = result[(result['date'] >= start_date) & (result['date'] <= end_date)]
        temp_lst = [['NYC', sum(result['Volume_gen_NYC'])/len(result)]]
        var_name = 'VolGenNYC_'+str(start_date.month)+str(start_date.day)+'_'+str(end_date.month)+str(end_date.day)
        result = pd.DataFrame(data = temp_lst, columns=['CTFIPS_1', var_name])
    return result




def ExtAttrByDays(county_OD_EXT):
    dat_lst = county_OD_EXT['date'].unique()
    result = []
    for dat in dat_lst:
        temp = county_OD_EXT[county_OD_EXT['date']==dat]
        temp = temp.groupby(['CTFIPS_2'])['Volume'].agg('sum').reset_index(name='Volume_atr')
        temp['date'] = dat
        result.append(temp)
        print(dat)
    result = pd.concat(result)
    result = result.groupby(['date'])['Volume_atr'].agg('sum').reset_index(name='Volume_atr')
    result['Volume_atr_thousands'] = (result['Volume_atr']/1000).round(0)
    result['Volume_atr_thousands_3dayavg'] = result.iloc[:, 2].rolling(window=3).mean()
    result['Volume_atr_thousands_3dayavg'] = (result['Volume_atr_thousands_3dayavg']).round(0)
    return result



def inputs_figure1_NYC():
    county_OD_EXT = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_OD_0515_cleaned_EXT.csv')
    county_OD_EXT['date'] = county_OD_EXT.apply(lambda z: pd.to_datetime(z.date), axis=1)
    jan2jan31 = ExtGen(county_OD_EXT, 'NYC', pd.Timestamp(date(2020,1,2)), pd.Timestamp(date(2020,1,31))) 
    mar2mar6 = ExtGen(county_OD_EXT, 'NYC', pd.Timestamp(date(2020,3,2)), pd.Timestamp(date(2020,3,6))) 
    mar16mar20 = ExtGen(county_OD_EXT, 'NYC', pd.Timestamp(date(2020,3,16)), pd.Timestamp(date(2020,3,20))) 
    mar30apr3 = ExtGen(county_OD_EXT, 'NYC', pd.Timestamp(date(2020,3,30)), pd.Timestamp(date(2020,4,3))) 
    apr13apr17 = ExtGen(county_OD_EXT, 'NYC', pd.Timestamp(date(2020,4,13)), pd.Timestamp(date(2020,4,17))) 
    apr27may1 = ExtGen(county_OD_EXT, 'NYC', pd.Timestamp(date(2020,4,27)), pd.Timestamp(date(2020,5,1))) 
    jan2jan31 = pd.merge(jan2jan31, mar2mar6, on= ['CTFIPS_1'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, mar16mar20, on= ['CTFIPS_1'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, mar30apr3, on= ['CTFIPS_1'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, apr13apr17, on= ['CTFIPS_1'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, apr27may1, on= ['CTFIPS_1'], how = 'left')
    jan2jan31.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_NYC.csv',index=False)
    return 

def inputs_figure1_ALL():
    county_OD_EXT = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_OD_0501_cleaned_EXT.csv')
    county_OD_EXT['date'] = county_OD_EXT.apply(lambda z: pd.to_datetime(z.date), axis=1)            
    jan2jan31 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,1,2)), pd.Timestamp(date(2020,1,31))) 
    mar2mar6 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,3,2)), pd.Timestamp(date(2020,3,6))) 
    mar16mar20 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,3,16)), pd.Timestamp(date(2020,3,20))) 
    mar30apr3 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,3,30)), pd.Timestamp(date(2020,4,3))) 
    apr13apr17 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,4,13)), pd.Timestamp(date(2020,4,17))) 
    apr27may1 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,4,27)), pd.Timestamp(date(2020,5,1))) 
    mar9mar13 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,3,9)), pd.Timestamp(date(2020,3,13))) 
    apr6apr10 = Avg_ExtAttr_AllCounties(county_OD_EXT, pd.Timestamp(date(2020,4,6)), pd.Timestamp(date(2020,4,10))) 
    jan2jan31 = pd.merge(jan2jan31, mar2mar6, on= ['CTFIPS_2'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, mar16mar20, on= ['CTFIPS_2'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, mar30apr3, on= ['CTFIPS_2'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, apr13apr17, on= ['CTFIPS_2'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, apr27may1, on= ['CTFIPS_2'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, mar9mar13, on= ['CTFIPS_2'], how = 'left')
    jan2jan31 = pd.merge(jan2jan31, apr6apr10, on= ['CTFIPS_2'], how = 'left')
    var_lst = ['VolAtr_316_320','VolAtr_413_417', 'VolAtr_427_51', 'VolAtr_39_313','VolAtr_46_410']
    for var in var_lst:
        new_var = 'C'+var
        jan2jan31[new_var] = ((jan2jan31[var] - jan2jan31['VolAtr_12_131']) / jan2jan31['VolAtr_12_131']*100).round(0)
        print(var)
    var_lst2 = [ 'C' + s for s in var_lst]    
    jan2jan31[['CTFIPS_2', 'VolAtr_12_131', 'CVolAtr_39_313', 'CVolAtr_46_410', 'CVolAtr_427_51', 'CVolAtr_316_320', 'CVolAtr_413_417']].to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_ALL.csv',index=False)
    return jan2jan31[['CTFIPS_2', 'VolAtr_12_131'] + var_lst2]

def NYC_OUT():
    county_OD_EXT = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_OD_0515_cleaned_EXT.csv')
    county_OD_EXT['CTFIPS_1'] = county_OD_EXT['CTFIPS_1'].astype('str').replace(['36061', '36047', '36005', '36081', '36085'], 'NYC')
    county_OD_EXT['CTFIPS_2'] = county_OD_EXT['CTFIPS_2'].astype('str').replace(['36061', '36047', '36005', '36081', '36085'], 'NYC')
    nyc_out = county_OD_EXT.groupby(['date','CTFIPS_1','CTFIPS_2'], as_index=False)['Volume'].agg('sum')
    nyc_out = nyc_out[nyc_out['CTFIPS_1'] != nyc_out['CTFIPS_2']]
    nyc_out = nyc_out[nyc_out['CTFIPS_1'] =='NYC']
    nyc_out['CTFIPS_2'] = nyc_out['CTFIPS_2'].astype('int')
    return nyc_out


def DestCentroidsVolavg(centroid,start_date, end_date):
    nyc_out = NYC_OUT()
    nyc_out['date'] = nyc_out.apply(lambda z: pd.to_datetime(z.date), axis=1)
    temp = nyc_out[(nyc_out['date'] >= start_date) & (nyc_out['date'] < end_date)]
    temp = temp.groupby(['CTFIPS_2'])['Volume'].agg('mean').reset_index(name='VolAvg_nycOut')
    result = pd.merge(temp, centroid, left_on='CTFIPS_2', right_on='CTFIPS', how='left')
    return result


def inputs_figure1_Dests():
    from datetime import date
    centroid = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/centroid_lat_long.csv', usecols=['CTFIPS','x','y']) # 18473
    jan2jan31 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,1,2)), pd.Timestamp(date(2020,1,31))) 
    jan2jan31.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_jan2jan31.csv',index=False)
    # mar2mar6 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,3,2)), pd.Timestamp(date(2020,3,6))) 
    mar9mar13 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,3,9)), pd.Timestamp(date(2020,3,13))) 
    mar9mar13.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_mar9mar13.csv',index=False)
    # mar30apr3 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,3,30)), pd.Timestamp(date(2020,4,3))) 
    apr6apr10 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,4,6)), pd.Timestamp(date(2020,4,10))) 
    apr6apr10.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_apr6apr10.csv',index=False)
    apr27may1 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,4,27)), pd.Timestamp(date(2020,5,1))) 
    apr27may1.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_apr27may1.csv',index=False)
    mar16mar20 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,3,16)), pd.Timestamp(date(2020,3,20))) 
    mar16mar20.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_mar16mar20.csv',index=False)
    apr13apr17 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,4,13)), pd.Timestamp(date(2020,4,17)))
    apr13apr17.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_apr13apr17.csv',index=False)
    # new 0515
    may11may15 = DestCentroidsVolavg(centroid, pd.Timestamp(date(2020,5,11)), pd.Timestamp(date(2020,5,15)))
    may11may15.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/inputs_figure1_Dests_may11may15.csv',index=False)    
    return     


def ranking_gen(level, start_date, end_date):  # level = ['daily','avg']
    county_OD_EXT = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_OD_0515_cleaned_EXT_NYC.csv')
    dat_lst = county_OD_EXT['date'].unique()
    result = []
    for dat in dat_lst:
        temp = county_OD_EXT[county_OD_EXT['date'] == dat]
        Num0fDests = temp.groupby(['CTFIPS_1']).size().reset_index(name='Num0fDests')
        Volume_ExtGen = temp.groupby(['CTFIPS_1'])['Volume'].agg('sum').reset_index(name='Volume_ExtGen')
        temp = pd.merge(Num0fDests, Volume_ExtGen, on = 'CTFIPS_1', how = 'inner')
        temp['date'] = dat
        temp['R_Num0fDests'] = temp['Num0fDests'].rank(ascending=False,method='min') 
        temp['R_Volume_ExtGen'] = temp['Volume_ExtGen'].rank(ascending=False,method='min') 
        result.append(temp)
    result = pd.concat(result)
    return result
nyc_gen = result[result['CTFIPS_1']=='NYC']

def ranking_attr(level, start_date, end_date):  # level = ['daily','avg']
    county_OD_EXT = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_OD_0515_cleaned_EXT_NYC.csv')
    dat_lst = county_OD_EXT['date'].unique()
    result = []
    for dat in dat_lst:
        temp = county_OD_EXT[county_OD_EXT['date'] == dat]
        Num0fOrigs = temp.groupby(['CTFIPS_2']).size().reset_index(name='Num0fOrigs')
        Volume_ExtAtr = temp.groupby(['CTFIPS_2'])['Volume'].agg('sum').reset_index(name='Volume_ExtAtr')
        temp = pd.merge(Num0fOrigs, Volume_ExtAtr, on = 'CTFIPS_2', how = 'inner')
        temp['date'] = dat
        temp['R_Num0fOrigs'] = temp['Num0fOrigs'].rank(ascending=False,method='min') 
        temp['R_Volume_ExtAtr'] = temp['Volume_ExtAtr'].rank(ascending=False,method='min') 
        result.append(temp)
    result = pd.concat(result)
    return result
nyc_atr = result[result['CTFIPS_2']=='NYC']
nyc_rank = pd.merge(nyc_gen, nyc_atr, on='date', how='inner')


import numpy as np
def getpercentiles(input_df):
    col_lst = input_df.columns
    result = []
    for col in col_lst:
        result = result + input_df[col].tolist()
        print(col)
    print('20th percentiles is %d' % np.nanpercentile(result, 20))
    print('40th percentiles is %d' % np.nanpercentile(result, 40))
    print('60th percentiles is %d' % np.nanpercentile(result, 60))
    print('80th percentiles is %d' % np.nanpercentile(result, 80))
    return
    
    
def input1_table():
    County_OD_0501_cleaned_EXT = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_OD_0501_cleaned_EXT.csv')
    County_OD_0501_cleaned_EXT['date'] = County_OD_0501_cleaned_EXT.apply(lambda z: pd.to_datetime(z.date), axis=1)
    County_OD_0501_cleaned_EXT['week'] = County_OD_0501_cleaned_EXT['date'].apply(lambda z: z.isocalendar()[1], axis=1)
    nycFips = [36061, 36047, 36005, 36081, 36085]
    County_OD_0501_cleaned_EXT['CTFIPS_1'] = County_OD_0501_cleaned_EXT['CTFIPS_1'].replace(nycFips, 'NYC')
    County_OD_0501_cleaned_EXT['CTFIPS_2'] = County_OD_0501_cleaned_EXT['CTFIPS_2'].replace(nycFips, 'NYC')
    County_OD_0501_cleaned_EXT = County_OD_0501_cleaned_EXT[County_OD_0501_cleaned_EXT['CTFIPS_1']!=County_OD_0501_cleaned_EXT['CTFIPS_2']]
    base_wk = [1,2,3,4,5]
    base = County_OD_0501_cleaned_EXT[County_OD_0501_cleaned_EXT['week'].isin(base_wk)]
    base = base.groupby(['date','CTFIPS_1'])['Volume'].agg('sum').reset_index(name='Volume_gen')
    base = base.groupby(['CTFIPS_1'])['Volume_gen'].agg('mean').reset_index(name='Volume_GenAvgJan')
    base['R_Volume_GenAvgJan'] = base['Volume_GenAvgJan'].rank()
    base['%_Volume_GenAvgJan'] = base['Volume_GenAvgJan']/sum(base['Volume_GenAvgJan'])
    other_wk = [6,7,8,9,10,11,12,13,14]
    other = County_OD_0501_cleaned_EXT[County_OD_0501_cleaned_EXT['week'].isin(other_wk)]
    result=[]
    for wk in other_wk:
        temp = other[other['week']==wk]
        temp = temp.groupby(['date','CTFIPS_1'])['Volume'].agg('sum').reset_index(name='Volume_gen')
        name1 = 'Volume_GenAvgWK'+str(wk)
        temp = temp.groupby(['CTFIPS_1'])['Volume_gen'].agg('mean').reset_index(name=name1)
        name2 = 'R_' + name1
        temp[name2] = temp[name1].rank()
        name3 = '%_' + name1
        temp[name3] = temp[name1]/sum(temp[name1])
        temp = temp[temp['CTFIPS_1']=='NYC']
        result.append(temp)
    result = pd.concat(result, axis=1)
    result = result.merge(base, on='CTFIPS_1', how='inner')
    return result


def input2_NYC_TopCounties(k):
    nyc_out = NYC_OUT()
    dat_lst = nyc_out['date'].unique().tolist()
    # dat_lst = dat_lst.tolist()
    lst = []
    for dat in dat_lst:
        temp = nyc_out[nyc_out['date']==dat]
        temp['rank'] = temp['Volume'].rank(ascending=False)
        temp = temp[temp['rank']<=k]
        temp = temp.sort_values(by=['rank'],ascending=True)
        ct_lst = temp['CTFIPS_2'].unique().tolist()
        lst=lst+ct_lst
    print(len(lst))
    lst = list(dict.fromkeys(lst))
    result = nyc_out[nyc_out['CTFIPS_2'].astype(int).isin(lst)]  
    # result.to_csv('H:/QQ/B_data/covid-19/external trips/outputs_2/input2_NYC_TopCounties_'+str(k)+'.csv',index=False)
    return [lst,result]
    
        
def input2_NYC_COVID(k): 
    lst_k = input2_NYC_TopCounties(k)[0]
    mobility_data = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_Raw_Results_0515.csv', usecols=['date','CTFIPS','CTNAME','STFIPS','New COVID cases'])
    mobility_data = mobility_data[mobility_data['CTFIPS']!='OutofUSA']
    mobility_data = mobility_data[mobility_data['CTFIPS'].astype(int).isin(lst_k)]    
    # 3-day moving average new cases
    # cumulative cases
    result=[]
    for ct in lst_k:
        temp = mobility_data[mobility_data['CTFIPS']==ct]
        temp['Cum_cases'] = temp['New COVID cases'].cumsum()
        temp['new_cases_3dayavg'] = temp['New COVID cases'].rolling(window=3).mean()
        result.append(temp)
    result = pd.concat(result)
    #result.to_csv('H:/QQ/B_data/covid-19/external trips/outputs_2/input2_NYC_TopCounties_'+str(k)+'_covid.csv',index=False)    
    return result


def save(df):
    df.to_csv('H:/QQ/B_data/covid-19/external trips/outputs/'+str(df)+'.csv',index=False)
    return
  
def ExtAtr_COVID_Dest0fNYC():
    nyc_out = NYC_OUT() 
    nyc_out['CTFIPS_2'] = nyc_out['CTFIPS_2'].astype(int)
    mobility_data = pd.read_csv('H:/QQ/B_data/covid-19/external trips/inputs/County_Raw_Results_0515.csv', usecols=['date','CTFIPS','New COVID cases','Population'])
    mobility_data = mobility_data[mobility_data['CTFIPS']!='OutofUSA']
    mobility_data['CTFIPS'] = mobility_data['CTFIPS'].astype(int)
    # filter by counties
    mobility_data = mobility_data[mobility_data['CTFIPS'].astype(int).isin(nyc_out['CTFIPS_2'].unique().tolist())]  
    # calculate cumulative cases for each county
    mobility_data['date'] = mobility_data.apply(lambda z: pd.to_datetime(z.date), axis=1)
    result = []
    for ct in nyc_out['CTFIPS_2'].unique().tolist():
        temp = mobility_data[mobility_data['CTFIPS']==ct]
        temp = temp.sort_values(by=['date'], ascending=True)
        temp['Cum_cases'] = temp['New COVID cases'].cumsum()
        result.append(temp)
    result = pd.concat(result)
    result = RM_weekends_holidays(result, holidays)
    result.rename(columns={'CTFIPS':'CTFIPS_2'},inplace=True)
    nyc_out['date'] = nyc_out.apply(lambda z: pd.to_datetime(z.date), axis=1)
    result['date'] = result.apply(lambda z: pd.to_datetime(z.date), axis=1)    
    nyc_out = nyc_out.merge(result, on=['date','CTFIPS_2'],how='left')
    nyc_out['Population'] = nyc_out['Population'].astype(float)
    nyc_out['NewCasesPerCapita']=nyc_out['New COVID cases']/nyc_out['Population']
    nyc_out['CumCasesPerCapita']=nyc_out['Cum_cases']/nyc_out['Population']
    nyc_out[['date','CTFIPS_2','Volume','New COVID cases','Cum_cases','NewCasesPerCapita','CumCasesPerCapita']].to_csv('H:/QQ/B_data/covid-19/external trips/outputs_2/ExtAtr_COVID_Dest0fNYC.csv',index=False)
    return nyc_out[['date','CTFIPS_2','Volume','NewCasesPerCapita','CumCasesPerCapita']]

def time_lag(start_date, end_date):  # start_date, end_date = pd.Timestamp(date(2020,3,9)), pd.Timestamp(date(2020,5,15))
    # df = ExtAtr_COVID_Dest0fNYC()   # df = nyc_out[['date','CTFIPS_2','Volume','NewCasesPerCapita','CumCasesPerCapita']]
    df = pd.read_csv('H:/QQ/B_data/covid-19/external trips/outputs_2/ExtAtr_COVID_Dest0fNYC.csv', usecols=['date','CTFIPS_2','Volume','NewCasesPerCapita','CumCasesPerCapita'])
    dat_idx = df[['date']].drop_duplicates().reset_index(drop=True)
    dat_idx['dat_idx'] = dat_idx.index
    df = df.merge(dat_idx,on='date',how='left')
    df['date'] = df.apply(lambda z: pd.to_datetime(z.date), axis=1) 
    df = df.sort_values(by=['date'],ascending=False)
    # df = nyc_out[['date','CTFIPS_2','Volume','NewCasesPerCapita','CumCasesPerCapita']]
    temp = df[(df['date']<=end_date) & (df['date']>=start_date)]
    dat_lst = temp['date'].astype(str).unique().tolist()
    result=[]
    for dat in dat_lst:
        temp=df[df['date']==dat]
        for n in [7,14,21]:
            dat = pd.to_datetime(dat)
            dat_2 = dat - np.timedelta64(n,'D')
            temp2 = df[df['date']==dat_2]
            if len(temp2)>0:
                temp2.rename(columns={'Volume':'Volume_'+str(n)+'daylag'},inplace=True)
                temp = temp.merge(temp2[['CTFIPS_2','Volume_'+str(n)+'daylag']], on=['CTFIPS_2'], how='left')
            print(dat,n)
        result.append(temp)
    result = pd.concat(result)
    result.rename(columns={'Volume':'Volume_0daylag'}, inplace=True)
    result.to_csv('H:/QQ/B_data/covid-19/external trips/outputs_2/ExtAtr_COVID_Dest0fNYC_withTimeLag_V3.csv',index=False)
    return

def corr_Pearson_Spearman(corr_type, lst1, lst2):
    if corr_type == 'Pearson':
        corr, _ = pearsonr(lst1, lst2)
    elif corr_type == 'Spearman':
        corr, _ = spearmanr(lst1, lst2)
    return corr


def correlation_NYC():
    df = pd.read_csv('H:/QQ/B_data/covid-19/external trips/outputs_2/ExtAtr_COVID_Dest0fNYC_withTimeLag_V3.csv')
    result=[]
    for dat in df['date'].unique().tolist():
        temp = df[df['date']==dat]
        for group in ['No time lag','One-week lag','Two-week lag','Three-week lag']:
            if group=='No time lag':
                n=0
            elif group == 'One-week lag':
                n=7
            elif group == 'Two-week lag':
                n=14
            elif group=='Three-week lag':
                n=21
            for case_type in ['NewCasesPerCapita','CumCasesPerCapita']:
                if case_type in temp.columns:
                    temp2 = temp[['Volume'+'_'+str(n)+'daylag',case_type]]
                    tlen = len(temp2)
                    temp2 = temp2.dropna()
                    # check outliers Q1-1.5IQR, Q3+1.5IQR; IQR = q3-q1
                    Q1 = np.nanpercentile(temp2['Volume'+'_'+str(n)+'daylag'], 25)
                    Q3 = np.nanpercentile(temp2['Volume'+'_'+str(n)+'daylag'], 75)
                    IQR = Q3-Q1
                    outlier = temp2[(temp2['Volume'+'_'+str(n)+'daylag']<Q1-1.5*IQR)|(temp2['Volume'+'_'+str(n)+'daylag']>Q3+1.5*IQR)]
                    temp2 =  temp2[(temp2['Volume'+'_'+str(n)+'daylag']>Q1-1.5*IQR) & (temp2['Volume'+'_'+str(n)+'daylag']<Q3+1.5*IQR)]
                    # print('Num0foutliers: ',len(outlier))
                    lst1 = temp2['Volume'+'_'+str(n)+'daylag'].tolist()
                    lst2 = temp2[case_type].tolist()
                    if (len(temp2)>=30):
                    #print(case_type)
                        for corr_type in ['Pearson','Spearman']:
                            value = corr_Pearson_Spearman(corr_type, lst1, lst2)
                            obs = len(temp2)
                            print('Valid obs: ',  len(temp2))
                            temp_lst = [dat,group,case_type,corr_type,value,obs]
                            result.append(temp_lst)
                            #print(dat,group,case_type,corr_type)
                    else:
                        print('Check ', dat,group,case_type)
    result = pd.DataFrame(data = result, columns=['date','group','case_type','corr_type','value','obs'])
    # moving average
    result['date'] = result.apply(lambda z: pd.to_datetime(z.date), axis=1)
    result = result.sort_values(by=['date'], ascending=True)
    result2=[]
    for group in  ['No time lag','One-week lag','Two-week lag','Three-week lag']:
        for case_type in ['NewCasesPerCapita','CumCasesPerCapita']:
            for corr_type in ['Pearson','Spearman']:
                temp = result[(result['group']==group) & (result['case_type']==case_type) & (result['corr_type']==corr_type)]
                temp['value_3dayavg'] = temp['value'].rolling(window=3).mean()
                result2.append(temp)
    result2 = pd.concat(result2)
    result2 = result2[result2['date']>=pd.Timestamp(date(2020,3,13))]
    return result2




def describe_corr(result2):
    result=[]
    for group in ['No time lag','One week lag','Two weeks lag','Three weeks lag']:
        for case_type in ['NewCasesPerCapita','CumCasesPerCapita']:
            for corr_type in ['Pearson','Spearman']:
                temp = result2[(result2['group']==group) & (result2['case_type']==case_type) & (result2['corr_type']==corr_type)]
                max_ = (temp['value_3dayavg']).max()
                mean = (temp['value_3dayavg']).mean()
                med = (temp['value_3dayavg']).median()
                result.append([group,case_type,corr_type,max_,mean,med])
    result = pd.DataFrame(data=result, columns=['group','case_type','corr_type','max_','mean','med'])
    return result

    