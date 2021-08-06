import pandas as pd
train=pd.read_csv('train.csv')
test=pd.read_csv('titanic_test.csv')
#test=test.loc[:,['2','4','5']]
train.info()#查看缺失值情况
#发现年龄有较少缺失值，而舱位有较多缺失值，去除舱位信息，补全年龄缺失值()
train.drop(labels='Cabin',axis=1,inplace=True)
train.drop('Ticket',axis=1,inplace=True)
test.fillna(test['5'].mean(),inplace=True)
train.fillna(train['Age'].mean(),inplace=True)
train.fillna(train['Embarked'].mode()[0],inplace=True)
print(test)
print(train)
train.info()
#贝叶斯分类器
def bayes(train_set,test_set):
    survivor=train_set[train_set['Survived']==1]
    victim=train_set[train_set['Survived']==0]
    sur_rate=len(survivor)/len(train_set)
    vic_rate=len(survivor)/len(train_set)
    #计算先验条件概率
    Pclass_rate1=len(survivor[survivor['Pclass']==1])/len(survivor)
    Pclass_rate2=len(survivor[survivor['Pclass']==2])/len(survivor)
    Pclass_rate3=len(survivor[survivor['Pclass']==3])/len(survivor)
    Pclass_rate1s = len(victim[victim['Pclass']== 1]) / len(victim)
    Pclass_rate2s = len(victim[victim['Pclass']== 2]) / len(victim)
    Pclass_rate3s = len(victim[victim['Pclass']== 3]) / len(victim)
    Sex_rate1=len(survivor[survivor['Sex']=='male'])/len(survivor)
    Sex_rate2 = len(survivor[survivor['Sex']=='female']) / len(survivor)
    Sex_rate1s = len(victim[victim['Sex']== 'male']) / len(victim)
    Sex_rate2s = len(victim[victim['Sex'] == 'female']) / len(victim)
    #print(type(survivor.itertuples()))#遍历Dataframe
    #    print(getattr(x,'Age'))
    #年龄的划分比较特殊，采用区间的方式划分
    Age_rate1=len([x for x in survivor.itertuples() if(getattr(x,'Age')>=0)and(getattr(x,'Age')<10)])/len(survivor)
    Age_rate2=len([x for x in survivor.itertuples() if(getattr(x,'Age')>=10)and(getattr(x,'Age')<40)])/len(survivor)
    Age_rate3 = len([x for x in survivor.itertuples() if(getattr(x,'Age')>=40)and(getattr(x,'Age')<60)]) / len(survivor)
    Age_rate4=len([x for x in survivor.itertuples() if(getattr(x,'Age')>=60)and(getattr(x,'Age')<100)])/len(survivor)
    Age_rate1s = len([x for x in victim.itertuples() if(getattr(x,'Age')>=0)and(getattr(x,'Age')<10)]) / len(victim)
    Age_rate2s = len([x for x in victim.itertuples() if(getattr(x,'Age')>=10)and(getattr(x,'Age')<40)]) / len(victim)
    Age_rate3s = len([x for x in victim.itertuples() if(getattr(x,'Age')>=40)and(getattr(x,'Age')<60)]) / len(victim)
    Age_rate4s = len([x for x in victim.itertuples() if(getattr(x,'Age')>=60)and(getattr(x,'Age')<100)]) / len(victim)
    #表达后验概率
    res=[]
    for index,x in test_set.iterrows():#遍历Dataframe
        if(x['2']==1):
            p=Pclass_rate1
        elif(x['2']==2):
            p=Pclass_rate2
        else:
            p=Pclass_rate3
        if(x['4']=='male'):
            p*=Sex_rate1
        else:
            p*=Sex_rate2
        if(x['5']>0 and x['5']<10):
            p*=Age_rate1
        elif(x['5']>=10 and x['5']<40):
            p*=Age_rate2
        elif(x['5']>=40 and x['5']>60):
            p*=Age_rate3
        else:
            p*=Age_rate4
        if (x['2'] == 1):
            p2 = Pclass_rate1s
        elif (x['2'] == 2):
            p2 = Pclass_rate2s
        else:
            p2 = Pclass_rate3s
        if (x['4'] == 'male'):
            p2 *= Sex_rate1s
        else:
            p2 *= Sex_rate2s
        if (x['5'] > 0 and x['5'] < 10):
            p2 *= Age_rate1s
        elif (x['5'] >= 10 and x['5'] < 40):
            p2 *= Age_rate2s
        elif (x['5'] >= 40 and x['5'] > 60):
            p2 *= Age_rate3s
        else:
            p2 *= Age_rate4s
        p*=sur_rate
        p2*=vic_rate
        if(p2<p):
            res.append('1')
        else:
            res.append('0')
    test_set['survive']=res
    return test_set
t=bayes(train,test)
t.to_excel('result.xls')