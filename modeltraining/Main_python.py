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


#如果遇到中文畫圖無法顯示字體
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默認字體
# mpl.RcParams['axes.unicode_minus'] = False # 解決保存圖片是負號"-"顯示為方塊問題
zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\msjh.ttc') #標楷體位置

#-----------------------------------------------------------------------------#
#                                                                             #
#                              IMK塑鉸參數區                                   #
#                                                                             #
#-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
#讀取EXCEL 
#當有sheet_name關鍵字參數的地方進行指定，如果有一個以上的話，組成串列(List)即可
# df = pd.read_excel("Taiwan RC Column Database.xlsx", sheet_name=["1", "2"])

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
    
    
df_pred_as = pd.DataFrame(pred_as, columns= ['as'])

df_Pred.iloc[:,[36]]= df_pred_as  #寫入格式    

print("----------------as測試結果----------------")
print('訓練集: ',randomForestModel_as.score(Xdata_train_Scale_as,Ydata_train_as).round(3))
#print('測試集: ',randomForestModel_as.score(Xdata_test_as,Ydata_test_as).round(3))
print("-----------------------------------------")




'''
#-----------------------------------------------------------------------------#
#預測c_S
#-----------------------------------------------------------------------------#
'''

Xdata_train_c_S =df_DataHub.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","緊密間距(cm)","fyt(MPa)","Av(cm2)","箍筋面積比ρt(%)","非緊密間距(cm)","as"]]
Ydata_train_c_S =  df_DataHub.loc[:,["c_S"]]


## 使用標準化Z-Score套件
z_score_scaler_X_c_S = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_c_S = z_score_scaler_X_c_S.fit_transform(Xdata_train_c_S)


#預測試體 
Xdata_test_c_S =df_Pred.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","緊密間距(cm)","fyt(MPa)","Av(cm2)","箍筋面積比ρt(%)","非緊密間距(cm)","as"]]
## 對數據進行標準化
Xdata_test_Scale_c_S = z_score_scaler_X_c_S.transform(Xdata_test_c_S)





# 建立RandomForestRegressor模型    n_estimators=200 ,max_depth= 10
randomForestModel_c_S = RandomForestRegressor(n_estimators=150 ,max_depth= 20,criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_c_S.fit(Xdata_train_Scale_c_S, np.ravel(Ydata_train_c_S))

# 使用訓練資料預測
predicted_c_S=randomForestModel_c_S.predict(Xdata_train_Scale_c_S).round(2)

# 使用測試資料預測   
pred_c_S = randomForestModel_c_S.predict(Xdata_test_Scale_c_S).round(2)
    
    
df_pred_c_S = pd.DataFrame(pred_c_S, columns= ['c_S'])

df_Pred.iloc[:,[37]]= df_pred_c_S  #寫入格式    

print("----------------c_S測試結果----------------")
print('訓練集: ',randomForestModel_c_S.score(Xdata_train_Scale_c_S,Ydata_train_c_S).round(3))
# print('測試集: ',randomForestModel_c_S.score(Xdata_test_c_S,Ydata_test_c_S).round(3))
print("-----------------------------------------")






'''
#-----------------------------------------------------------------------------#
#預測c_K
#-----------------------------------------------------------------------------#
'''

Xdata_train_c_K =df_DataHub.loc[:,["混凝土強度f'c(kgf/cm^2)","保護層(cm)","柱高(cm)","鋼筋斷面積As","Av(cm2)","箍筋面積比ρt(%)","破壞形式"]]
Ydata_train_c_K =  df_DataHub.loc[:,["c_K"]]


## 使用標準化Z-Score套件
z_score_scaler_X_c_K = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_c_K = z_score_scaler_X_c_K.fit_transform(Xdata_train_c_K)


#預測試體
Xdata_test_c_K =df_Pred.loc[:,["混凝土強度f'c(kgf/cm^2)","保護層(cm)","柱高(cm)","鋼筋斷面積As","Av(cm2)","箍筋面積比ρt(%)","破壞形式"]]
## 對數據進行標準化
Xdata_test_Scale_c_K = z_score_scaler_X_c_K.transform(Xdata_test_c_K)



# 建立RandomForestRegressor模型
randomForestModel_c_K = RandomForestRegressor(n_estimators=250,max_depth=25, criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_c_K.fit(Xdata_train_Scale_c_K, np.ravel(Ydata_train_c_K))
    
# 使用訓練資料預測
predicted_c_K=randomForestModel_c_K.predict(Xdata_train_Scale_c_K).round(2)
# 使用測試資料預測     
pred_c_K = randomForestModel_c_K.predict(Xdata_test_Scale_c_K).round(2)
    
    
df_pred_c_K = pd.DataFrame(pred_c_K, columns= ['c_K'])

df_Pred.iloc[:,[40]]= df_pred_c_K  #寫入格式         


print("----------------c_K測試結果----------------")
print('訓練集: ',randomForestModel_c_K.score(Xdata_train_Scale_c_K,Ydata_train_c_K).round(3))
#print('測試集: ',randomForestModel_c_K.score(Xdata_test_c_K,Ydata_test_c_K).round(3))
print("-----------------------------------------")




'''
#-----------------------------------------------------------------------------#
#預測c_A
#-----------------------------------------------------------------------------#
'''

Xdata_train_c_A = df_DataHub.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","as","c_K","非緊密間距(cm)"]]
Ydata_train_c_A =  df_DataHub.loc[:,["c_A"]]

## 使用標準化Z-Score套件
z_score_scaler_X_c_A = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_c_A = z_score_scaler_X_c_A.fit_transform(Xdata_train_c_A)


#預測試體
Xdata_test_c_A =df_Pred.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","as","c_K","非緊密間距(cm)"]]
## 對數據進行標準化
Xdata_test_Scale_c_A = z_score_scaler_X_c_A.transform(Xdata_test_c_A)



# 建立RandomForestRegressor模型   n_estimators=100,max_depth=15,
randomForestModel_c_A = RandomForestRegressor(n_estimators=150,max_depth=19,criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_c_A.fit(Xdata_train_Scale_c_A, np.ravel(Ydata_train_c_A))

# 使用訓練資料預測
predicted_c_A=randomForestModel_c_A.predict(Xdata_train_Scale_c_A).round(2)
# 使用測試資料預測   
pred_c_A = randomForestModel_c_A.predict(Xdata_test_Scale_c_A).round(2)
    
    
df_pred_c_A = pd.DataFrame(pred_c_A, columns= ['c_A'])

df_Pred.iloc[:,[39]]= df_pred_c_A  #寫入格式      


print("----------------c_A測試結果----------------")
print('訓練集: ',randomForestModel_c_A.score(Xdata_train_Scale_c_A,Ydata_train_c_A).round(3))
#print('測試集: ',randomForestModel_c_A.score(Xdata_test_c_A,Ydata_test_c_A).round(3))
print("-----------------------------------------")


    

'''
#-----------------------------------------------------------------------------#    
#預測θp
#-----------------------------------------------------------------------------#
'''
#"破壞形式","混凝土強度f'c(kgf/cm^2)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_C","c_K","c_A","θpc"
Xdata_train_θp =df_DataHub.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_K","c_A"]]
Ydata_train_θp =  df_DataHub.loc[:,["θp"]]

## 使用標準化Z-Score套件
z_score_scaler_X_θp = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_θp = z_score_scaler_X_θp.fit_transform(Xdata_train_θp)


#預測試體
Xdata_test_θp =df_Pred.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_K","c_A"]]
## 對數據進行標準化
Xdata_test_Scale_θp = z_score_scaler_X_θp.transform(Xdata_test_θp)



# 建立RandomForestRegressor模型 n_estimators=100,max_depth=17 
randomForestModel_θp = RandomForestRegressor( n_estimators=50,max_depth=15,criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_θp.fit(Xdata_train_Scale_θp, np.ravel(Ydata_train_θp))

# 使用訓練資料預測
predicted_θp=randomForestModel_θp.predict(Xdata_train_Scale_θp).round(4)
# 使用測試資料預測 
pred_θp = randomForestModel_θp.predict(Xdata_test_Scale_θp).round(4)
    
    
df_pred_θp = pd.DataFrame(pred_θp, columns= ['θp'])

df_Pred.iloc[:,[41]]= df_pred_θp  #寫入格式      
    
    

print("----------------θp測試結果----------------")
print('訓練集: ',randomForestModel_θp.score(Xdata_train_Scale_θp,Ydata_train_θp).round(3))
#print('測試集: ',randomForestModel_θp.score(Xdata_test_θp,Ydata_test_θp).round(3))
print("-----------------------------------------")



'''
#-----------------------------------------------------------------------------#
#預測c_C
#-----------------------------------------------------------------------------#
'''

Xdata_train_c_C =df_DataHub.loc[:,["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","Av(cm2)","箍筋面積比ρt(%)","θp","破壞形式","非緊密間距(cm)"]]
Ydata_train_c_C =  df_DataHub.loc[:,["c_C"]]


## 使用標準化Z-Score套件
z_score_scaler_X_c_C = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_c_C = z_score_scaler_X_c_C.fit_transform(Xdata_train_c_C)


#預測試體
Xdata_test_c_C =df_Pred.loc[:,["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","Av(cm2)","箍筋面積比ρt(%)","θp","破壞形式","非緊密間距(cm)"]]
## 對數據進行標準化
Xdata_test_Scale_c_C = z_score_scaler_X_c_C.transform(Xdata_test_c_C)





# 建立RandomForestRegressor模型 n_estimators=850 ,max_depth= 13, 
randomForestModel_c_C = RandomForestRegressor(n_estimators=45 ,max_depth= 20,criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_c_C.fit(Xdata_train_Scale_c_C, np.ravel(Ydata_train_c_C))

# 使用訓練資料預測
predicted_c_C=randomForestModel_c_C.predict(Xdata_train_Scale_c_C).round(2)

# 使用測試資料預測
pred_c_C = randomForestModel_c_C.predict(Xdata_test_Scale_c_C).round(2)
    
    
df_pred_c_C = pd.DataFrame(pred_c_C, columns= ['c_C'])

df_Pred.iloc[:,[38]]= df_pred_c_C  #寫入格式      
    

print("----------------c_C測試結果----------------")
print('訓練集: ',randomForestModel_c_C.score(Xdata_train_Scale_c_C,Ydata_train_c_C).round(3))
#print('測試集: ',randomForestModel_c_C.score(Xdata_test_c_C,Ydata_test_c_C).round(3))
print("-----------------------------------------")




    


'''
#-----------------------------------------------------------------------------#    
#預測θpc
#-----------------------------------------------------------------------------#    
'''


Xdata_train_θpc =df_DataHub.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_C","c_K","c_A","θp"]]
Ydata_train_θpc =  df_DataHub.loc[:,["θpc"]]

## 使用標準化Z-Score套件
z_score_scaler_X_θpc = StandardScaler()

## 對數據進行標準化
Xdata_train_Scale_θpc = z_score_scaler_X_θpc.fit_transform(Xdata_train_θpc)


#預測試體
Xdata_test_θpc =df_Pred.loc[:,["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_C","c_K","c_A","θp"]]
## 對數據進行標準化
Xdata_test_Scale_θpc = z_score_scaler_X_θpc.transform(Xdata_test_θpc)







# 建立RandomForestRegressor模型 n_estimators=100,max_depth=12,
randomForestModel_θpc = RandomForestRegressor(n_estimators=1000,max_depth=24,criterion = 'squared_error',max_features = "sqrt" )
# 使用訓練資料訓練模型
randomForestModel_θpc.fit(Xdata_train_Scale_θpc, np.ravel(Ydata_train_θpc))

# 使用訓練資料預測
predicted_θpc=randomForestModel_θpc.predict(Xdata_train_Scale_θpc).round(4)       
# 使用測試資料預測
pred_θpc = randomForestModel_θpc.predict(Xdata_test_Scale_θpc).round(4)
    
    
df_pred_θpc = pd.DataFrame(pred_θpc, columns= ['θpc'])

df_Pred.iloc[:,[42]]= df_pred_θpc  #寫入格式      
    
            
            

print("----------------θpc測試結果----------------")
print('訓練集: ',randomForestModel_θpc.score(Xdata_train_Scale_θpc,Ydata_train_θpc).round(3))
#print('測試集: ',randomForestModel_θpc.score(Xdata_test_θpc,Ydata_test_θpc).round(3))
print("-----------------------------------------")


df_Pred.iloc[:,[43]]= 0


#輸出為EXCEL
# df_Pred.to_excel('預測.xlsx',sheet_name='預測值',index=False)
df_newData = pd.concat([df_addNew,df_Pred],axis = 0)
df_newData.reset_index(drop = True ,inplace = True)
df_newData.to_excel('新試體資料.xlsx',sheet_name='過往預測',index=False)

print("----------------預測完成,數據輸出完成----------------")

Pred = {"a_s ":pred_as,
            "c_S":pred_c_S,
            "c_K":pred_c_K,
            "c_A":pred_c_A,
            "θp":pred_θp,
            "c_C":pred_c_C,
            "θpc":pred_θpc }
print(Pred)

import pickle
import gzip
with gzip.GzipFile('./model_pkl/randomForestModel_as.pgz', 'w') as f:
 pickle.dump(randomForestModel_as, f)
with gzip.GzipFile('./model_pkl/randomForestModel_c_S.pgz', 'w') as f:
 pickle.dump(randomForestModel_c_S, f)
with gzip.GzipFile('./model_pkl/randomForestModel_c_K.pgz', 'w') as f:
 pickle.dump(randomForestModel_c_K, f)
with gzip.GzipFile('./model_pkl/randomForestModel_c_A.pgz', 'w') as f:
 pickle.dump(randomForestModel_c_A, f)
with gzip.GzipFile('./model_pkl/randomForestModel_θp.pgz', 'w') as f:
 pickle.dump(randomForestModel_θp, f)
with gzip.GzipFile('./model_pkl/randomForestModel_c_C.pgz', 'w') as f:
 pickle.dump(randomForestModel_c_C, f)
with gzip.GzipFile('./model_pkl/randomForestModel_θpc.pgz', 'w') as f:
 pickle.dump(randomForestModel_θpc, f)

with gzip.GzipFile("./model_pkl/z_score_scaler_X_as.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_as, f)
with gzip.GzipFile("./model_pkl/z_score_scaler_X_c_S.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_c_S, f)
with gzip.GzipFile("./model_pkl/z_score_scaler_X_c_K.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_c_K, f)
with gzip.GzipFile("./model_pkl/z_score_scaler_X_c_A.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_c_A, f)
with gzip.GzipFile("./model_pkl/z_score_scaler_X_θp.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_θp, f)
with gzip.GzipFile("./model_pkl/z_score_scaler_X_c_C.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_c_C, f)
with gzip.GzipFile("./model_pkl/z_score_scaler_X_θpc.pgz", "w") as f:
 pickle.dump(z_score_scaler_X_θpc, f)



 