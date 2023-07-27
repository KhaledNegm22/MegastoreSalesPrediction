
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dython.nominal import associations
from dython.nominal import identify_nominal_columns
import re
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as colors
import squarify
from sklearn.preprocessing import MinMaxScaler
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import seaborn as sns
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# In[137]:


data = pd.read_csv("megastore-regression-dataset.csv")


# In[304]:


analyzeddf = pd.read_csv("megastore-regression-dataset.csv")


# In[138]:


data


# In[139]:


data.describe()


# In[140]:


data.info()


# In[141]:


for column in data.columns:
    print(column + ";", "Number of null values:",data[column].isna().sum())


# In[142]:


for column in data.columns:
    print(column+";", "Number of Unique Values:",len(data[column].unique()))


# In[143]:


categorical_features = identify_nominal_columns(data)
categorical_features


# In[145]:


df=data


# # Visualization

# In[13]:


df_bar = df[['Region','Profit']]

df_bar = df_bar.groupby('Region').mean().sort_values(by='Profit', ascending=False)

plt.figure(figsize=[15,8])

plt.suptitle("Average Profit Across Different Regions", size=20)

plt.subplot(1,2,1)
plt.bar(x = df_bar.index,
        height = 'Profit',
        color=['#3C486B','#F45050','#F9D949','#DDDDDD'],
        data = df_bar)
plt.show()


# In[14]:


df_bar = df[['Ship Mode','Profit']]

df_bar = df_bar.groupby('Ship Mode').mean().sort_values(by='Profit', ascending=False)

plt.figure(figsize=[15,8])

plt.suptitle("Average Profit Across Different Ship Mode", size=20)

plt.subplot(1,2,1)

plt.bar(x=df_bar.index,
        height='Profit',
        color=['#3C486B','#F45050','#F9D949','#DDDDDD'],
        data=df_bar)

plt.show()


# In[15]:


df_tree = df[['Segment','Profit']]

df_tree = df_tree.groupby(['Segment']).sum().reset_index()

plt.figure(figsize=[10,8])

squarify.plot(sizes=df_tree['Profit'], label = df_tree['Segment'],
              color=['#3C486B','#F45050','#F9D949'], alpha=0.7)

plt.title("Profit Across Different Segment", size=20, pad=20)

plt.axis('off')

plt.show()


# ### Checking Correlation between Variables

# In[305]:


associations(analyzeddf,
             mark_columns=True,
             nom_nom_assoc='theil',
             num_num_assoc='pearson',
             figsize=(25,25),
             title='Correlation Between all variables')


# ### Preprocessing

# In[147]:


df


# ### Preprocessing of order date and ship data by cyclical encoding and get the differnece between ship date and order date

# In[148]:


df['Ship Date'] = pd.to_datetime(data['Ship Date'])
df['Order Date'] = pd.to_datetime(data['Order Date'])


# In[149]:


df['delayDays'] = (df['Ship Date'] - df['Order Date']).dt.days


# In[150]:


df['Ship month'] = df['Ship Date'].dt.month
df['Ship day'] = df['Ship Date'].dt.day
df['Ship year'] = df['Ship Date'].dt.year


# In[151]:


df['Order month'] = df['Order Date'].dt.month
df['Order day'] = df['Order Date'].dt.day
df['Order year'] = df['Order Date'].dt.year


# In[152]:


def cyclicalEncode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


# In[153]:


cyclicalEncode(df,'Ship month',12)
cyclicalEncode(df,'Ship day',31)
cyclicalEncode(df,'Order month',12)
cyclicalEncode(df,'Order day',31)


# In[154]:


ax = df.plot.scatter('Ship month_sin', 'Ship month_cos').set_aspect('equal')


# In[155]:


ax = df.plot.scatter('Ship day_sin', 'Ship day_cos').set_aspect('equal')


# In[156]:


ax = df.plot.scatter('Order month_sin', 'Order month_cos').set_aspect('equal')


# In[157]:


ax = df.plot.scatter('Order day_sin', 'Order day_cos').set_aspect('equal')


# ### Encoding Ship mode

# In[158]:


df['Ship Mode'].unique()


# In[159]:


def encodeShipMode():
    rownum = -1
    for i in df['Ship Mode']:
        rownum += 1
        if(i == 'First Class'):
            df.iloc[rownum, -1] = 4
        elif(i == 'Standard Class'):
            df.iloc[rownum, -1] = 3
        elif(i == 'Second Class'):
            df.iloc[rownum, -1] = 2
        elif(i == 'Same Day'):
            df.iloc[rownum, -1] = 1
        else:
            df.iloc[rownum, -1] = 0


# In[160]:


df['encoded ship mode'] = df['Ship Mode']


# In[161]:


encodeShipMode()


# ### Encode Customer ID - Name

# ##### Proving that the letters part in the Customer id is the first letter from first name and last name of Customer Name

# ##### Detecting the names that have more than two names in it to double verify the assumption

# In[163]:


ans = 0
rownum = -1
isIdentical = True
for i in data['Customer Name']:
    x = i.split()
    rownum += 1
    if(len(x) > 2):
        ans += 1
        CID = str(df.iloc[rownum, 4])
        print(i, "  ",CID)
        if(str(x[0][0]) != CID[0] and str(x[1][0]) != CID[1]):
            isIdentical = False
print("count of names that have more than 2 names :" , ans)
print(isIdentical)


# In[164]:


df[['CustomerIDLetter', 'CustomerIDNumber']] = df['Customer ID'].str.split("-", expand = True)


# In[165]:


len(df['CustomerIDNumber'].unique()) == len(data['Customer Name'].unique())


# In[166]:


enc = preprocessing.OrdinalEncoder()


# In[167]:


namereshaped = np.array(df['Customer Name']).reshape(-1,1)


# In[168]:


df['Labeled Customer names'] = enc.fit_transform(namereshaped)


# In[169]:


df.info()


# In[170]:


df


# In[171]:


df['encoded ship mode'] = df['encoded ship mode'].astype(str).astype(int)


# In[172]:


df['CustomerIDNumber'] = df['CustomerIDNumber'].astype(str).astype(int)


# ### Encode Segment

# In[173]:


df['Segment'].unique()


# In[174]:


df['labeled Segment'] = df['Segment']


# In[175]:


def encodeSegment():
    rownum = -1
    for i in df['Segment']:
        rownum += 1
        if(i == 'Consumer'):
            df.iloc[rownum, -1] = 3
        elif(i == 'Corporate'):
            df.iloc[rownum, -1] = 2
        elif(i == 'Home Office'):
            df.iloc[rownum, -1] = 1
        else:
            df.iloc[rownum, -1] = 0


# In[176]:


encodeSegment()


# In[177]:


df['labeled Segment'] = df['labeled Segment'].astype(str).astype(int)


# ### Encode Country - City - State

# In[178]:


countryreshaped = np.array(df['Country']).reshape(-1,1)
df['Labeled Country'] = enc.fit_transform(countryreshaped)


# In[179]:


cityreshaped = np.array(df['City']).reshape(-1,1)
df['Labeled City'] = enc.fit_transform(cityreshaped)


# In[180]:


statereshaped = np.array(df['State']).reshape(-1,1)
df['Labeled state'] = enc.fit_transform(statereshaped)


# ### Encode Region

# In[181]:


df['labeled Region'] = df['Region']


# In[182]:


def encodeRegion():
    rownum = -1
    for i in df['Region']:
        rownum += 1
        if(i == 'East'):
            df.iloc[rownum, -1] = 4
        elif(i == 'West'):
            df.iloc[rownum, -1] = 3
        elif(i == 'South'):
            df.iloc[rownum, -1] = 2
        elif(i == 'Central'):
            df.iloc[rownum, -1] = 1
        else:
            df.iloc[rownum, -1] = 0


# In[183]:


encodeRegion()


# In[184]:


df['labeled Region'] = df['labeled Region'].astype(str).astype(int)


# In[185]:


df['Region'].unique()


# ### Encode Product ID

# In[186]:


df[['productmain', 'productsub','productnum']] = df['Product ID'].str.split("-", expand = True)


# In[187]:


productmainreshaped = np.array(df['productmain']).reshape(-1,1)
df['productmain'] = enc.fit_transform(productmainreshaped)


# In[188]:


productsubreshaped = np.array(df['productsub']).reshape(-1,1)
df['productsub'] = enc.fit_transform(productsubreshaped)


# In[189]:


df['productmain'] = df['productmain'].astype(float).astype(int)
df['productsub'] = df['productsub'].astype(float).astype(int)


# In[190]:


df['Final Product ID'] = df['productmain'].astype(str) + df['productsub'].astype(str)


# In[191]:


df['Final Product ID'] = df['Final Product ID'].astype(str) + df['productnum'].astype(str)


# In[192]:


df['productnum'] = df['productnum'].astype(str).astype(int)


# ### Encode Category Tree

# In[193]:


df[['MainCategory', 'SubCategory']] = df['CategoryTree'].str.split(",", expand = True)


# In[194]:


CategoryTree = "{'MainCategory': 'Technology', 'SubCategory': 'Phones'}"
MainCategory = re.search(r"MainCategory': '([\w\s]+)'", CategoryTree).group(1)
SubCategory = re.search(r"SubCategory': '([\w\s]+)'", CategoryTree).group(1)
print(MainCategory)
print(SubCategory)


# In[195]:


rownum = -1
for i in df['MainCategory']:
    rownum += 1
    df.iloc[rownum, -2] = re.search(r"MainCategory': '([\w\s]+)'", i).group(1)


# In[196]:


rownum = -1
for i in df['SubCategory']:
    rownum += 1
    df.iloc[rownum, -1] = re.search(r"SubCategory': '([\w\s]+)'", i).group(1)


# In[197]:


productmainreshaped = np.array(df['productmain']).reshape(-1,1)
df['productmain'] = enc.fit_transform(productmainreshaped)


# In[198]:


df['labeled MainCategory'] = df['MainCategory']


# In[199]:


df['labeled SubCategory'] = df['SubCategory']


# In[200]:


df


# In[201]:


mainctegoryreshaped = np.array(df['labeled MainCategory']).reshape(-1,1)
df['labeled MainCategory'] = enc.fit_transform(mainctegoryreshaped)


# In[202]:


subctegoryreshaped = np.array(df['labeled SubCategory']).reshape(-1,1)
df['labeled SubCategory'] = enc.fit_transform(subctegoryreshaped)


# In[203]:


len(df['Product Name'].unique())


# In[204]:


productnamereshaped = np.array(df['Product Name']).reshape(-1,1)
df['Product Name'] = enc.fit_transform(productnamereshaped)


# In[205]:


df[['orderidcountry', 'orderidyear','orderidnumber']] = df['Order ID'].str.split("-", expand = True)


# In[206]:


orderidcountryreshaped = np.array(df['orderidcountry']).reshape(-1,1)
df['orderidcountry'] = enc.fit_transform(orderidcountryreshaped)


# In[207]:


df['Order ID'] = df['orderidcountry'].astype(str) + df['orderidyear'].astype(str) + df['orderidnumber'].astype(str)


# In[208]:


df['Order ID'] = df['Order ID'].astype(str).astype(float)


# In[209]:


df.drop(columns=['Row ID','Order Date','Ship Date','Ship Mode','Customer ID','Customer Name',
                'Segment','Country','City','State','Region','Product ID','CategoryTree'],inplace=True)


# In[210]:


df.describe()


# In[211]:


df.drop(columns=['MainCategory','SubCategory'],inplace=True)


# In[212]:


df.drop(columns=['orderidcountry','orderidyear','orderidnumber'],inplace=True)


# In[213]:


df.info()


# In[214]:


df['Final Product ID'] = df['Final Product ID'].astype(str).astype(float)


# In[215]:


df.drop(columns=['CustomerIDLetter'],inplace=True)


# In[216]:


data.info()


# In[217]:


Y = df['Profit']


# In[218]:


X = df.drop(columns=['Profit'])


# In[219]:


scaler = MinMaxScaler()


# In[220]:


scaledPostal = np.array(df['Postal Code']).reshape(-1,1)
df['Postal Code'] = scaler.fit_transform(scaledPostal)


# In[221]:


scaledProductID = np.array(df['Final Product ID']).reshape(-1,1)
df['Final Product ID'] = scaler.fit_transform(scaledProductID)


# In[222]:


scaledCustomerID = np.array(df['CustomerIDNumber']).reshape(-1,1)
df['CustomerIDNumber'] = scaler.fit_transform(scaledCustomerID)


# In[223]:


scaledsales = np.array(df['Sales']).reshape(-1,1)
df['Sales'] = scaler.fit_transform(scaledsales)


# In[226]:


df.info()


# In[227]:


df.drop(columns=['Order month'],inplace=True)


# In[228]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,random_state=42, shuffle=False)


# In[229]:


df.info()


# ## Random Forrest Model

# In[237]:


model = sfs(RandomForestRegressor(),
            k_features = 10,
            forward = True,
            verbose = 2,
            cv = 5,
            n_jobs = -1,
            scoring='r2')
model.fit(X_train,y_train)


# In[253]:


model.k_feature_names_


# In[230]:


XRandomForrest_train = X_train[['Sales',
 'Discount',
 'encoded ship mode',
 'labeled Segment',
 'Labeled Country',
 'productmain',
 'productsub']]


# In[231]:


modelRandomForrest = RandomForestRegressor().fit(XRandomForrest_train,y_train)


# In[232]:


y_pred_RandomForrest = modelRandomForrest.predict(XRandomForrest_train)


# In[285]:


print('R2 Score training of Random Forrest', metrics.r2_score(y_train, y_pred_RandomForrest) * 100)


# In[313]:


sns.regplot(x=y_train, y = y_pred_RandomForrest,
           scatter_kws={"color": "blue"}, line_kws={"color": "red"})

plt.show()
# In[234]:


XRandomForrest_test = X_test[['Sales',
 'Discount',
 'encoded ship mode',
 'labeled Segment',
 'Labeled Country',
 'productmain',
 'productsub']]


# In[235]:


y_pred_RandomForrest_test = modelRandomForrest.predict(XRandomForrest_test)


# In[284]:


print('R2 Score of test Random Forrest', metrics.r2_score(y_test, y_pred_RandomForrest_test) * 100)


# In[312]:


sns.regplot(x=y_test, y = y_pred_RandomForrest_test,
           scatter_kws={"color": "blue"}, line_kws={"color": "red"})


# ## XGBoost Regression Model

# In[114]:


best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                                      gamma=0,
                                      learning_rate=0.07,
                                      max_depth=3,
                                      min_child_weight = 1.5,
                                      n_estimators=10000,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)


# In[240]:


xgboostmodel = sfs(best_xgb_model,
            k_features = 10,
            forward = True,
            verbose = 2,
            cv = 5,
            n_jobs = -1,
            scoring='r2')
xgboostmodel.fit(X_train,y_train)


# In[242]:


xgboostmodel.k_feature_names_


# In[237]:


Xxgboost_train = X_train[['Sales',
 'Discount',
 'Ship day_sin',
 'Ship day_cos',
 'Labeled Country',
 'labeled Region',
 'productmain',
 'productsub']]


# In[238]:


modelXGboost = best_xgb_model.fit(Xxgboost_train ,y_train)


# In[239]:


y_pred_Xgboost = modelXGboost.predict(Xxgboost_train)


# In[286]:


print('R2 Score training of XGboost', metrics.r2_score(y_train, y_pred_Xgboost) * 100)


# In[311]:


sns.regplot(x=y_train, y = y_pred_Xgboost,
           scatter_kws={"color": "blue"}, line_kws={"color": "red"})


# In[241]:


Xxgboost_test = X_test[['Sales',
 'Discount',
 'Ship day_sin',
 'Ship day_cos',
 'Labeled Country',
 'labeled Region',
 'productmain',
 'productsub']]


# In[242]:


y_pred_Xgboost_test = modelXGboost.predict(Xxgboost_test)


# In[287]:


print('R2 Score of test XGboost', metrics.r2_score(y_test, y_pred_Xgboost_test) * 100)


# In[310]:


sns.regplot(x=y_test, y = y_pred_Xgboost_test,
           scatter_kws={"color": "blue"}, line_kws={"color": "red"})

plt.show()
# ## Polynomial Regression Model

# In[244]:


poly_features = PolynomialFeatures(degree = 3)


# In[268]:


X_train_poly = poly_features.fit_transform(X_train)
X_train_polycols = poly_features.get_feature_names_out(X_train.columns)
X_test_poly = poly_features.fit_transform(X_test)
X_test_polycols = poly_features.get_feature_names_out(X_test.columns)


# In[269]:


x_train_poly_df = pd.DataFrame(X_train_poly, columns = X_train_polycols)


# In[270]:


X_test_poly_df = pd.DataFrame(X_test_poly, columns = X_test_polycols)


# In[271]:


x_train_poly_df.columns


# In[272]:


X_test_poly_df.columns


# In[254]:


poly_model = LinearRegression()


# In[273]:


polynomialmodel = sfs(poly_model,
            k_features = 10,
            forward = True,
            verbose = 2,
            cv = 5,
            n_jobs = -1,
            scoring='r2')
polynomialmodel.fit(x_train_poly_df, y_train)


# In[274]:


polynomialmodel.k_feature_names_


# In[275]:


X_trian_polynomial = x_train_poly_df[['Sales productnum',
 'Sales Discount^2',
 'Sales Ship month productmain',
 'Sales Order month Final Product ID',
 'Sales Order year encoded ship mode',
 'Sales Ship month_cos Labeled Customer names',
 'Sales Ship day_cos Order month_sin',
 'Sales encoded ship mode productsub',
 'Sales labeled Region productmain',
 'Sales productsub^2']]


# In[281]:


X_test_polynomial = X_test_poly_df[['Sales productnum',
 'Sales Discount^2',
 'Sales Ship month productmain',
 'Sales Order month Final Product ID',
 'Sales Order year encoded ship mode',
 'Sales Ship month_cos Labeled Customer names',
 'Sales Ship day_cos Order month_sin',
 'Sales encoded ship mode productsub',
 'Sales labeled Region productmain',
 'Sales productsub^2']]


# In[276]:


poly_model.fit(X_trian_polynomial, y_train)


# In[279]:


y_train_predicted_Polynomial = poly_model.predict(X_trian_polynomial)


# In[289]:


print('R2 Score of Polynomial Train', metrics.r2_score(y_train, y_train_predicted_Polynomial) * 100)


# In[309]:


sns.regplot(x=y_train, y = y_train_predicted_Polynomial,
           scatter_kws={"color": "blue"}, line_kws={"color": "red"})

plt.show()
# In[282]:


y_test_Predicted_Polynomial = poly_model.predict(X_test_polynomial)


# In[315]:


print('R2 Score of Polynomial Test', metrics.r2_score(y_test, y_test_Predicted_Polynomial) * 100)


# In[314]:


sns.regplot(x=y_test, y = y_test_Predicted_Polynomial,
            scatter_kws={"color": "blue"}, line_kws={"color": "red"})


# In[ ]:




