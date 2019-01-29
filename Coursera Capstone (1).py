
# coding: utf-8

# ### ETL

# In[1]:


import types
import pandas as pd
from ibm_botocore.client import Config
import ibm_boto3
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import keras as K
from keras.layers import Dense, Dropout,LSTM,Flatten
from keras import optimizers


# In[2]:



#Fetch data from object storage and store in a dict of pandas dataframes



def __iter__(self): return 0

dfDict = {}

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_14cc115e76e94f3290186e32865f4ee6 = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='RhEptNVmX6CZ4LltjVcp_FzUZMztxbVWfgX5rIye2v2T',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Uk_FTSE100.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Uk = pd.read_csv(body)
df_data_Uk.head()
dfDict['Uk'] = df_data_Uk

body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Sp_IBEX35.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Sp = pd.read_csv(body)
df_data_Sp.head()
dfDict['Sp'] = df_data_Sp


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Si_StratsTimes.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Si = pd.read_csv(body)
df_data_Si.head()
dfDict['Si'] = df_data_Si


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Na_NASDAQComp.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_NASDAQ = pd.read_csv(body)
df_data_NASDAQ.head()
dfDict['NASDAQ'] = df_data_NASDAQ


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Na_DowJonesIndustrial.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_DOW = pd.read_csv(body)
df_data_DOW.head()
dfDict['DOW'] = df_data_DOW


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Me_IPC.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Me = pd.read_csv(body)
df_data_Me.head()
dfDict['Me'] = df_data_Me


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Jp_NIKKEI225.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Jp = pd.read_csv(body)
df_data_Jp.head()
dfDict['Jp'] = df_data_Jp


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='In_Sensex.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_In = pd.read_csv(body)
df_data_In.head()
dfDict['In'] = df_data_In


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Hk_HangSeng.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Hk = pd.read_csv(body)
df_data_Hk.head()
dfDict['Hk'] = df_data_Hk


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Gr_DAX.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Gr = pd.read_csv(body)
df_data_Gr.head()
dfDict['Gr'] = df_data_Gr


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Fr_CAC40.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Fr = pd.read_csv(body)
df_data_Fr.head()
dfDict['Fr'] = df_data_Fr


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Ch_ShanghaiComp.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Ch = pd.read_csv(body)
df_data_Ch.head()
dfDict['Ch'] = df_data_Ch


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Ca_SP.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Ca = pd.read_csv(body)
df_data_Ca.head()
dfDict['Ca'] = df_data_Ca


body = client_14cc115e76e94f3290186e32865f4ee6.get_object(Bucket='default-donotdelete-pr-oql892xhwlhuar',Key='Au_ASX200.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_Au = pd.read_csv(body)
df_data_Au.head()
dfDict['Au'] = df_data_Au



# In[3]:



dfCopy = dfDict.copy()

#add a column for the labels and drop unnecessary columns
for df in dfCopy.values():
    
    #create a column for the label
    df['Label'] = 0
    
    #drop useless columns and drop data such that each dataframe can be divided in batches of 100 evenly
    df.drop([' Open',' High',' Low'],axis =1,inplace = True)
    df.drop(df.index[0:df.shape[0]%60],inplace = True)
    df.reset_index(inplace = True ,drop = True)
    
    #normalise each data set individually
    scaler = MinMaxScaler(copy = True, feature_range = (0,1))
    dfScaled = scaler.fit_transform(df[' Close'].reshape(-1,1))
    df[' Close'] = dfScaled
    #df.set_index('Date',inplace = True)
    
    
print(dfDict)   


# In[4]:


#extract individual dataframes from dict

dfAu = dfDict['Au']
dfAu['Country'] = 'Au'

dfCa = dfDict['Ca']
dfCa['Country'] = 'Ca'

dfCh = dfDict['Ch']
dfCh['Country'] = 'Ch'

dfFr = dfDict['Fr']
dfFr['Country'] = 'Fr'

dfGr = dfDict['Gr']
dfGr['Country'] = 'Gr'

dfHk = dfDict['Hk']
dfHk['Country'] = 'Hk'

dfIn = dfDict['In']
dfIn['Country'] = 'In'

dfJp = dfDict['Jp']
dfJp['Country'] = 'Jp'

dfMe = dfDict['Me']
dfMe['Country'] = 'Me'

dfDOW = dfDict['DOW']
dfDOW['Country'] = 'DOW'

dfNASDAQ = dfDict['NASDAQ']
dfNASDAQ['Country'] = 'NASDAQ'

dfSi = dfDict['Si']
dfSi['Country'] = 'Si'

dfSp = dfDict['Sp']
dfSp['Country'] = 'Sp'

dfUk = dfDict['Uk']
dfUk['Country'] = 'Uk'


#label stock market crashes as 1's useing dates from https://en.wikipedia.org/wiki/List_of_stock_market_crashes_and_bear_markets

#2015–16 stock market selloff USA
loc = dfDOW.loc[dfDOW['Date']=='06/16/15'].index.values[0]
dfDOW['Label'][loc:loc + 60] = 1

loc = dfNASDAQ.loc[dfNASDAQ['Date']=='06/16/15'].index.values[0]
dfNASDAQ['Label'][loc:loc + 60] = 1


#2015–16 Chinese stock market crash
loc = dfCh.loc[dfCh['Date']=='06/12/15'].index.values[0]
dfCh['Label'][loc:loc + 60] = 1


#2010 Flash Crash USA

loc = dfDOW.loc[dfDOW['Date']=='05/06/10'].index.values[0]
dfDOW['Label'][loc:loc + 60] = 1

loc = dfNASDAQ.loc[dfNASDAQ['Date']=='05/06/10'].index.values[0]
dfNASDAQ['Label'][loc:loc + 60] = 1


#European sovereign debt crisis

loc = dfUk.loc[dfUk['Date']=='10/04/10'].index.values[0]
dfUk['Label'][loc:loc + 60] = 1

loc = dfSp.loc[dfSp['Date']=='10/04/10'].index.values[0]
dfSp['Label'][loc:loc + 60] = 1

loc = dfGr.loc[dfGr['Date']=='10/04/10'].index.values[0]
dfGr['Label'][loc:loc + 60] = 1

loc = dfFr.loc[dfFr['Date']=='10/04/10'].index.values[0]
dfFr['Label'][loc:loc + 60] = 1




# In[5]:


#Financial crisis of 2007–08

loc = dfAu.loc[dfAu['Date']=='09/16/08'].index.values[0]
dfAu['Label'][loc:loc + 60] = 1

loc = dfCa.loc[dfCa['Date']=='09/16/08'].index.values[0]
dfCa['Label'][loc:loc + 60] = 1

loc = dfCh.loc[dfCh['Date']=='09/16/08'].index.values[0]
dfCh['Label'][loc:loc + 60] = 1

loc = dfFr.loc[dfFr['Date']=='09/16/08'].index.values[0]
dfFr['Label'][loc:loc + 60] = 1

loc = dfGr.loc[dfGr['Date']=='09/16/08'].index.values[0]
dfGr['Label'][loc:loc + 60] = 1

loc = dfHk.loc[dfHk['Date']=='09/16/08'].index.values[0]
dfHk['Label'][loc:loc + 60] = 1

loc = dfIn.loc[dfIn['Date']=='09/16/08'].index.values[0]
dfIn['Label'][loc:loc + 60] = 1

loc = dfJp.loc[dfJp['Date']=='09/16/08'].index.values[0]
dfJp['Label'][loc:loc + 60] = 1

loc = dfSi.loc[dfSi['Date']=='09/16/08'].index.values[0]
dfSi['Label'][loc:loc + 60] = 1

loc = dfSp.loc[dfSp['Date']=='09/16/08'].index.values[0]
dfSp['Label'][loc:loc + 60] = 1

loc = dfUk.loc[dfUk['Date']=='09/16/08'].index.values[0]
dfUk['Label'][loc:loc + 60] = 1

loc = dfDOW.loc[dfDOW['Date']=='09/16/08'].index.values[0]
dfDOW['Label'][loc:loc + 60] = 1

loc = dfNASDAQ.loc[dfNASDAQ['Date']=='09/16/08'].index.values[0]
dfNASDAQ['Label'][loc:loc + 60] = 1


# In[6]:


#United States bear market of 2007–09

loc = dfDOW.loc[dfDOW['Date']=='10/11/07'].index.values[0]
dfDOW['Label'][loc:loc + 60] = 1

loc = dfNASDAQ.loc[dfNASDAQ['Date']=='10/11/07'].index.values[0]
dfNASDAQ['Label'][loc:loc + 60] = 1


#Chinese stock bubble of 2007

loc = dfCh.loc[dfCh['Date']=='02/26/07'].index.values[0]
dfCh['Label'][loc:loc + 60] = 1


#Stock market downturn of 2002

loc = dfDOW.loc[dfDOW['Date']=='10/09/02'].index.values[0]
dfDOW['Label'][loc:loc + 60] = 1

loc = dfNASDAQ.loc[dfNASDAQ['Date']=='10/09/02'].index.values[0]
dfNASDAQ['Label'][loc:loc + 60] = 1

loc = dfUk.loc[dfUk['Date']=='10/09/02'].index.values[0]
dfUk['Label'][loc:loc + 60] = 1

loc = dfGr.loc[dfGr['Date']=='10/09/02'].index.values[0]
dfUk['Label'][loc:loc + 60] = 1

loc = dfFr.loc[dfFr['Date']=='10/09/02'].index.values[0]
dfFr['Label'][loc:loc + 60] = 1

loc = dfAu.loc[dfAu['Date']=='10/09/02'].index.values[0]
dfAu['Label'][loc:loc + 60] = 1

loc = dfCa.loc[dfCa['Date']=='10/09/02'].index.values[0]
dfCa['Label'][loc:loc + 60] = 1

loc = dfSp.loc[dfSp['Date']=='10/09/02'].index.values[0]
dfSp['Label'][loc:loc + 60] = 1


#Dot-com bubble

loc = dfDOW.loc[dfDOW['Date']=='03/10/00'].index.values[0]
dfDOW['Label'][loc:loc + 60] = 1

loc = dfNASDAQ.loc[dfNASDAQ['Date']=='03/10/00'].index.values[0]
dfNASDAQ['Label'][loc:loc + 60] = 1

#1997 Asian financial crisis

loc = dfCh.loc[dfCh['Date']=='07/02/97'].index.values[0]
dfCh['Label'][loc:loc + 60] = 1


# In[7]:


get_ipython().magic(u'matplotlib inline')
dfUk.plot(y =' Close',x ='Date')
dfUk.plot(y ='Label',x ='Date')


# In[8]:


totalDF = pd.DataFrame()

for df in dfDict.values():
    totalDF = totalDF.append(df)


X = totalDF[' Close'].values
Ya = totalDF['Label'].values
Y=[]

Ydummy = np.array_split(Ya,Ya.shape[0]/60)
for y in Ydummy:
    if y.sum() != 0:
        Y.append(1)
    else:
        Y.append(0)


X = X.reshape(int(X.size/60),60,1)
print(X.shape)


# In[9]:


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


# In[18]:



model = K.Sequential()
model.add(LSTM(10,stateful = False, activation = 'relu',return_sequences=True,input_shape = (60,1)))
model.add(Dropout(0.2))
model.add(LSTM(20,stateful = False, activation = 'relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(20,stateful = False, activation = 'relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid',name = 'output'))
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])


# In[19]:


model.fit(X_train, Y_train, batch_size=30, epochs=15, validation_data = (X_test, Y_test), shuffle=False)


# In[ ]:


#save model
model_json = model.to_json()
with open("StockLSTM.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("StockLSTM.h5")


# In[17]:


#Evaluate the model 

scores = model.evaluate(X_test, Y_test, verbose=1)
print(scores)


# In[25]:


res = model.predict(X_train,batch_size = 30)


# In[26]:


res.max()

