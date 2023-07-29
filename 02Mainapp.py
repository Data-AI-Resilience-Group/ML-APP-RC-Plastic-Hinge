#-----------------------------------------------------------------------
# 匯入套件
#-----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import datetime

base="light"
secondaryBackgroundColor="#b1a9a9"
textColor="#000000"
font="serif"


st.write("""
# Prediction app

This app predicts 
         
""")

#-----------------------------------------------------------------------
# sidebar 使用者輸入特徵
#-----------------------------------------------------------------------

st.sidebar.write('''
                 
# User Input Features

''')

failuremode_raw = st.sidebar.selectbox('Column failure mode',('Shear failure','Flexural failure'))
Specimen_Name = st.sidebar.text_input("Specimen Name")
fc = st.sidebar.number_input("Compressive strength of concrete f'c (kgf/cm^2)",step=0.0,value=100.0,max_value=1000.0,min_value=0.01)
concretecover = st.sidebar.number_input('Concrete cover(cm)')
Axialratio = st.sidebar.number_input('Axial load ratio')
Ag = st.sidebar.number_input('Cross-sectional area of the column')
columnheight = st.sidebar.number_input("Height of the column")
fy = st.sidebar.number_input('Yield strength of main reinforcement fy(kgf/cm)')
arearatio = st.sidebar.number_input('Main reinforcement area ratio ρL(%)')
As = st.sidebar.number_input('Cross-sectional area of the steel reinforcement')
tightspacing = st.sidebar.number_input('Tight spacing(cm)')
widespacing = st.sidebar.number_input("Wide spacing(cm)")
fyt = st.sidebar.number_input('fyt(Mpa)')
hook = st.sidebar.number_input('Hook angle')
Av = st.sidebar.number_input('Av(cm2)')
At = st.sidebar.number_input("Shear reinforcement ratio ρt(%)")


if failuremode_raw == 'Shear failure':
    failuremode = 0
else:
  failuremode_raw == 'Flexural failure'
  failuremode = 1

data = {"Compressive strength of concrete f'c (kgf/cm^2)": fc,
        'Concrete cover(cm)': concretecover,
        'Cross-sectional area of the column': Ag,
        'Height of the column':columnheight ,
        'Yield strength of main reinforcement fy(kgf/cm)':fy ,
        'fyt(Mpa)':fyt ,
        'Main reinforcement area ratio ρL(%)':arearatio ,
        'Shear reinforcement ratio ρt(%)': At,
        'Cross-sectional area of the steel reinforcement':As ,
        'Av(cm2)': Av,
        'Tight spacing(cm)':tightspacing ,
        'Wide spacing(cm)': widespacing,
        'Column failure mode': failuremode,
        'axial load ratio':Axialratio ,
        'Hook angle':hook }

features = pd.DataFrame(data, index=[0])



#-----------------------------------------------------------------------
# 按下Submit後開始預測，匯入個參數之隨機森林模型
#-----------------------------------------------------------------------

if st.sidebar.button("Submit"):
    

    import pickle
    import gzip

    with gzip.open('./randomForestModel_as.pgz', 'r') as f:
     randomForestModel_as = pickle.load(f)
    with gzip.open('./randomForestModel_c_S.pgz', 'r') as f:
     randomForestModel_c_S = pickle.load(f)
    with gzip.open('./randomForestModel_c_K.pgz', 'r') as f:
     randomForestModel_c_K = pickle.load(f)
    with gzip.open('./randomForestModel_c_A.pgz', 'r') as f:
     randomForestModel_c_A = pickle.load(f)
    with gzip.open('./randomForestModel_θp.pgz', 'r') as f:
     randomForestModel_θp = pickle.load(f)
    with gzip.open('./randomForestModel_c_C.pgz', 'r') as f:
     randomForestModel_c_C = pickle.load(f)
    with gzip.open('./randomForestModel_θpc.pgz', 'r') as f:
     randomForestModel_θpc = pickle.load(f)

    with gzip.open("z_score_scaler_X_as.pgz", "r") as f:
     z_score_scaler_X_as = pickle.load(f)
    with gzip.open("z_score_scaler_X_c_S.pgz", "r") as f:
     z_score_scaler_X_c_S = pickle.load(f)
    with gzip.open("z_score_scaler_X_c_K.pgz", "r") as f:
     z_score_scaler_X_c_K = pickle.load(f)
    with gzip.open("z_score_scaler_X_c_A.pgz", "r") as f:
     z_score_scaler_X_c_A = pickle.load(f)
    with gzip.open("z_score_scaler_X_θp.pgz", "r") as f:
     z_score_scaler_X_θp = pickle.load(f)
    with gzip.open("z_score_scaler_X_c_C.pgz", "r") as f:
     z_score_scaler_X_c_C = pickle.load(f)
    with gzip.open("z_score_scaler_X_θpc.pgz", "r") as f:
     z_score_scaler_X_θpc = pickle.load(f)




    df_TW = pd.read_excel("Taiwan RC Column Database.xlsx", sheet_name="Base" ,skiprows=1)
    df_CN = pd.read_excel("PRJ-2793.xlsx", sheet_name="Base" ,skiprows=1)
    df_peer_F = pd.read_excel("PEER RC Column Database(Rectangular).xlsx", sheet_name="Flexure" ,skiprows=1)
    df_peer_FS = pd.read_excel("PEER RC Column Database(Rectangular).xlsx", sheet_name="FlexureShear" ,skiprows=1)
    df_peer_S = pd.read_excel("PEER RC Column Database(Rectangular).xlsx", sheet_name="Shear" ,skiprows=1)
    df_addNew = pd.read_excel("app新試體資料.xlsx", sheet_name="過往預測" )
    df_DataHub = pd.concat([df_TW,df_peer_FS,df_peer_S,df_peer_F,df_CN,df_addNew],axis = 0)
    df_DataHub .reset_index(drop = True ,inplace = True)
    df_Pred = pd.read_excel("輸入模型資料.xlsx", sheet_name="AI塑鉸預測" ,skiprows=1)

    #-----------------------------------------------------------------------
    ### 匯入使用者輸入資料至模型 ###
    #-----------------------------------------------------------------------
    
    df_Pred.iloc[:,[0]] = Specimen_Name  #試體名稱

    current_datetime = datetime.datetime.now()
    df_Pred.iloc[:,[1]] = current_datetime #時間

    df_Pred.iloc[:,[5]] = fc

    df_Pred.iloc[:,[6]] = concretecover

    df_Pred.iloc[:,[14]] = Ag

    df_Pred.iloc[:,[15]] = columnheight

    df_Pred.iloc[:,[20]] = fy

    df_Pred.iloc[:,[29]] = fyt

    df_Pred.iloc[:,[22]] = arearatio

    df_Pred.iloc[:,[34]] = At

    df_Pred.iloc[:,[23]] = As

    df_Pred.iloc[:,[33]] = Av

    df_Pred.iloc[:,[27]] = tightspacing

    df_Pred.iloc[:,[28]] = widespacing

    df_Pred.iloc[:,[3]] = failuremode

    df_Pred.iloc[:,[8]] = Axialratio

    df_Pred.iloc[:,[32]] = hook

    #-----------------------------------------------------------------------
    #預測as
    #-----------------------------------------------------------------------

    Xdata_test_as= pd.DataFrame([[fc, concretecover, Axialratio, Ag, columnheight, fy, arearatio, As, tightspacing, At, failuremode, widespacing]], 
             columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","箍筋面積比ρt(%)","破壞形式","非緊密間距(cm)"])

    # 使用標準化Z-Score套件
    # 對數據進行標準化 

    Xdata_test_Scale_as = z_score_scaler_X_as.transform(Xdata_test_as)

    # 使用測試資料預測  
    pred_as_ = randomForestModel_as.predict(Xdata_test_Scale_as).round(4)
    df_pred_as = pd.DataFrame(pred_as_, columns= ['as'])
        
    df_Pred.iloc[:,[36]]= df_pred_as  #寫入格式    
    #-----------------------------------------------------------------------
    #預測c_S
    #-----------------------------------------------------------------------

    Xdata_test_c_S = pd.DataFrame([[failuremode, fc, concretecover, Axialratio, Ag, columnheight, fy, arearatio, tightspacing, fyt, Av, At, widespacing, pred_as_]], 
                              columns=["破壞形式", "混凝土強度f'c(kgf/cm^2)", "保護層(cm)", "軸力比", "斷面積(cm2)", "柱高(cm)", "降伏強度fy(kgf/cm)", "主筋面積比ρL(%)", "緊密間距(cm)", "fyt(MPa)", "Av(cm2)", "箍筋面積比ρt(%)", "非緊密間距(cm)", "as"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_S = z_score_scaler_X_c_S.transform(Xdata_test_c_S)

     
    # 使用測試資料預測 
    pred_c_S = randomForestModel_c_S.predict(Xdata_test_Scale_c_S).round(2)
    df_pred_c_S = pd.DataFrame(pred_c_S, columns= ['c_S'])
        
    df_Pred.iloc[:,[37]]= df_pred_c_S  #寫入格式    
    #-----------------------------------------------------------------------
    #預測c_K
    #-----------------------------------------------------------------------

    Xdata_test_c_K= pd.DataFrame([[fc, concretecover, columnheight, As, Av, At, failuremode]], 
                     columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","柱高(cm)","鋼筋斷面積As","Av(cm2)","箍筋面積比ρt(%)","破壞形式"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_K = z_score_scaler_X_c_K.transform(Xdata_test_c_K)

    
    # 使用測試資料預測 
    pred_c_K = randomForestModel_c_K.predict(Xdata_test_Scale_c_K).round(4)
    df_pred_c_K = pd.DataFrame(pred_c_K, columns= ['c_K'])
        
    df_Pred.iloc[:,[40]]= df_pred_c_K  #寫入格式      
    #-----------------------------------------------------------------------
    #預測c_A
    #-----------------------------------------------------------------------

    Xdata_test_c_A= pd.DataFrame([[failuremode, fc, concretecover, Axialratio, Ag, columnheight, fy, arearatio, As, tightspacing, fyt, hook,Av, At, pred_as_, pred_c_K , widespacing]], 
                     columns = ["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","as","c_K","非緊密間距(cm)"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_A = z_score_scaler_X_c_A.transform(Xdata_test_c_A)

    
    # 使用測試資料預測 
    pred_c_A = randomForestModel_c_A.predict(Xdata_test_Scale_c_A).round(4)
    df_pred_c_A = pd.DataFrame(pred_c_A, columns= ['c_A'])
        
    df_Pred.iloc[:,[39]]= df_pred_c_A  #寫入格式     
    #-----------------------------------------------------------------------
    #預測θp
    #-----------------------------------------------------------------------

    Xdata_test_θp= pd.DataFrame([[failuremode, fc, concretecover, Axialratio, Ag, columnheight, fy, arearatio, As, tightspacing, widespacing, fyt, hook, Av, At, pred_c_S, pred_as_, pred_c_K ,pred_c_A ]], 
                     columns = ["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_K","c_A"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_θp = z_score_scaler_X_θp.transform(Xdata_test_θp)

    
    # 使用測試資料預測 
    pred_θp = randomForestModel_θp.predict(Xdata_test_Scale_θp).round(4)
    df_pred_θp = pd.DataFrame(pred_θp, columns= ['θp'])
        
    df_Pred.iloc[:,[41]]= df_pred_θp  #寫入格式    
    #-----------------------------------------------------------------------
    #預測c_C
    #-----------------------------------------------------------------------

    Xdata_test_c_C= pd.DataFrame([[fc, concretecover, Axialratio, Ag, columnheight, fy, arearatio, As, tightspacing, fyt, Av, At, pred_θp, failuremode, widespacing]], 
                     columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","Av(cm2)","箍筋面積比ρt(%)","θp","破壞形式","非緊密間距(cm)"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_C = z_score_scaler_X_c_C.transform(Xdata_test_c_C)

    
    # 使用測試資料預測 
    pred_c_C = randomForestModel_c_C.predict(Xdata_test_Scale_c_C).round(4)
    df_pred_c_C = pd.DataFrame(pred_c_C, columns= ['c_C'])
        
    df_Pred.iloc[:,[38]]= df_pred_c_C  #寫入格式      
    #-----------------------------------------------------------------------
    #預測_θpc
    #-----------------------------------------------------------------------

    Xdata_test_θpc= pd.DataFrame([[failuremode, fc, concretecover, Axialratio, Ag, columnheight, fy, arearatio, As, tightspacing, widespacing, fyt, hook, Av, At, pred_c_S, pred_as_, pred_c_C, pred_c_K ,pred_c_A, pred_θp]], 
                     columns = ["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_C","c_K","c_A","θp"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_θpc = z_score_scaler_X_θpc.transform(Xdata_test_θpc)

    
    # 使用測試資料預測 
    pred_θpc = randomForestModel_θpc.predict(Xdata_test_Scale_θpc).round(4)

    df_pred_θpc = pd.DataFrame(pred_θpc, columns= ['θpc'])
        
    df_Pred.iloc[:,[42]]= df_pred_θpc  #寫入格式      

    #輸出為EXCEL
    # df_Pred.to_excel('預測.xlsx',sheet_name='預測值',index=False)
    df_newData = pd.concat([df_addNew,df_Pred],axis = 0)
    df_newData.reset_index(drop = True ,inplace = True)
    df_newData.to_excel('app新試體資料.xlsx',sheet_name='過往預測',index=False)

    #-----------------------------------------------------------------------
    ### 預測完成，預測結果匯出 ###
    #-----------------------------------------------------------------------

    st.subheader('User Input features')
    st.write(features) #顯示輸入特徵

    st.subheader('Prediction')
    Pred = {"a_s ":pred_as_,
                  "c_S":pred_c_S,
                  "c_K":pred_c_K,
                  "c_A":pred_c_A,
                  "θp":pred_θp,
                  "c_C":pred_c_C,
                  "θpc":pred_θpc }
    Prediction = pd.DataFrame(Pred, index=[0]) 
    st.write(Prediction) #顯示預測結果


    #-----------------------------------------------------------------------
    #開啟網頁終端機輸入:streamlit run 02Mainapp.py  
    #-----------------------------------------------------------------------