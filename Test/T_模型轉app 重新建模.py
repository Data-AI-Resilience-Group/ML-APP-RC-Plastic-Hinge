import numpy as np
import pandas as pd 

#----------------------------------------------------------------------
#machine learning use
import matplotlib
#切割訓練集、交叉驗證
from sklearn.model_selection import train_test_split,cross_val_score
#隨機森林
from sklearn.ensemble import RandomForestRegressor

## 導入sklearn 標準化套件
from sklearn.preprocessing import StandardScaler

df_TW = pd.read_excel("Taiwan RC Column Database.xlsx", sheet_name="Base" ,skiprows=1)
df_CN = pd.read_excel("PRJ-2793.xlsx", sheet_name="Base" ,skiprows=1)
#,skiprows=1
df_peer_F = pd.read_excel("PEER RC Column Database(Rectangular).xlsx", sheet_name="Flexure" ,skiprows=1)
df_peer_FS = pd.read_excel("PEER RC Column Database(Rectangular).xlsx", sheet_name="FlexureShear" ,skiprows=1)
df_peer_S = pd.read_excel("PEER RC Column Database(Rectangular).xlsx", sheet_name="Shear" ,skiprows=1)
df_addNew = pd.read_excel("新試體資料.xlsx", sheet_name="過往預測" )
#-----------------------------------------------------------------------------#
#iat[資料索引值,欄位順序]：利用資料索引值及欄位順序來取得「單一值」
df_DataHub = pd.concat([df_TW,df_peer_FS,df_peer_S,df_peer_F,df_CN,df_addNew],axis = 0)
# df_DataHub = pd.concat([df_TW,df_peer_FS,df_peer_S,df_peer_F,df_addNew],axis = 0)
df_DataHub .reset_index(drop = True ,inplace = True)

df_Pred = pd.read_excel("輸入模型資料.xlsx", sheet_name="AI塑鉸預測" ,skiprows=1)



print("----------------預測開始----------------")
"""
將全部數據當成訓練集去預測一個試體
"""
'''
# #-----------------------------------------------------------------------------#
# #預測as
# #-----------------------------------------------------------------------------#
# '''

Xdata_train_as =df_DataHub.loc[:,["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","箍筋面積比ρt(%)","破壞形式","非緊密間距(cm)"]]
Ydata_train_as =  df_DataHub.loc[:,["as"]]

## 使用標準化Z-Score套件
z_score_scaler_X_as = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_as = z_score_scaler_X_as.fit_transform(Xdata_train_as)


# ## 轉換成DataFrame
# Xdata_z_score = pd.DataFrame(scale_data)

fc=280
concretecover=6.35 
Axialratio=0.01789
Ag=3600
columnheight=427
fy=4696
As=108.36
ρ=As/Ag
tightspacing=10
ρ_sh=0.31
failuremode=1
widespacing=10

#預測試體
Xdata_test_as =df_Pred.loc[:,["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","箍筋面積比ρt(%)","破壞形式","非緊密間距(cm)"]]
## 對數據進行標準化
Xdata_test_Scale_as = z_score_scaler_X_as.transform(Xdata_test_as)


# 建立RandomForestRegressor模型 n_estimators=1000 ,max_depth= 10
randomForestModel_as = RandomForestRegressor( n_estimators=200 ,max_depth= 21,criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_as.fit(Xdata_train_Scale_as, np.ravel(Ydata_train_as))
    
# 使用訓練資料預測 
predicted_as=randomForestModel_as.predict(Xdata_train_Scale_as).round(4)


# 建立測試集的 DataFrme
# 使用測試資料預測       
pred_as = randomForestModel_as.predict(Xdata_test_Scale_as).round(4)

print(Xdata_test_as)
print(pred_as)