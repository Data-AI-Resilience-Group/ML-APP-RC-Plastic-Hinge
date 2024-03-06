#-----------------------------------------------------------------------
## 匯入套件
#-----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import datetime
from github import Github
import os
import math
import pickle
import gzip

# 設定 Streamlit 應用程式的外觀

st.set_page_config(
    page_title="Streamlit App",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# st.image("picture/DataAI.jpg", caption="", use_column_width=True)
st.write("""
         
# AI Plastic Hinge Prediction app


This application harnesses a comprehensive dataset of cyclic loading experiments conducted on 475 reinforced concrete columns.
Using numerical simulation and analysis through OpenSees, we've synthesized key geometric and material properties.
These properties serve as inputs for predicting nonlinear plastic hinge spring parameters within the modified Ibarra-Medina-Krawinkler deterioration model (IMK model). The prediction process employs a data-driven model that combines Active Learning and Random Forest methodologies. Users can effortlessly provide input information as per the outlined description, and the application will subsequently generate the 7 essential parameters required for the IMK model.
         
---
         
""")


#-----------------------------------------------------------------------
# sidebar 使用者輸入特徵
#-----------------------------------------------------------------------


st.sidebar.write("""
                 
# User Input Features
               
Please enter the parameters of the specimen **described in the legend below**, ensuring **non-zero values** and **appropriate units** are used during input.

---

## Basic information""")


Specimen_Name = st.sidebar.text_input("Specimen Name")
fc = st.sidebar.text_input("""Compressive strength of **concrete** f'c (kgf/cm^2)""",value=0)
fy = st.sidebar.text_input('''Yield stress of **longitudinal reinforcement fy(kgf/cm)**''',value=0)
fy_sh = st.sidebar.text_input('''Yield stress of **transverse reinforcement fy,sh(Mpa)**''',value=0)

st.sidebar.write("""
                  
---
                                    
## Cross-sectional dimension""")
local_image_path = "picture/crosssection.jpg"  # 替換為你自己的本地圖片路徑
st.sidebar.image(local_image_path, caption="Cross section", use_column_width=True)

b = st.sidebar.text_input('width of column cross-section(cm)',value=0)
d = st.sidebar.text_input('''**depth of column cross-section(cm)**, measured from the extreme compressive fiber to the centerline of tensile reinforcemens''',value=0)
h = st.sidebar.text_input('Height of column cross-section(cm)',value=0)
c = st.sidebar.text_input('Concrete cover(cm)',value=0)
H = st.sidebar.text_input("Height of the column (cm)",value=0)

st.sidebar.write("""
---

## Reinforcement
                 
                 """)
hook = st.sidebar.text_input('Hook angle(°)',value=0)
As = st.sidebar.text_input('Area of steel rebar m^2(c)',value=0)
A_sh = st.sidebar.text_input('Area of shear reinforcement(cm^2)',value=0)

local_image_path = "picture/Column.jpg"  # 替換為你自己的本地圖片路徑
st.sidebar.image(local_image_path, caption="Column shear reinforcement arrangement", use_column_width=True)

tightspacing = st.sidebar.text_input('Tight spacing(cm) TS',value=0)
non_tightspacing = st.sidebar.text_input("Non-tight spacing(cm) NS",value=0)


st.sidebar.write("""
---
## External forces
                 """)
P = st.sidebar.text_input('Axial load(tf)',value=0)
Mn = st.sidebar.text_input('Maximum moment capacity in model(tf-m)',value=0)

st.sidebar.write("""
---
                 """)



#-----------------------------------------------------------------------
# 按下Submit後開始預測，匯入個參數之隨機森林模型
#-----------------------------------------------------------------------

if st.sidebar.button("Submit"):

    #-----------------------------------------------------------------------
    # 基本參數計算
    #-----------------------------------------------------------------------
    b = float(b)
    d = float(d)
    h = float(h)
    P = float(P)
    fc = float(fc)
    As =float(As)
    A_sh = float(A_sh)
    fy_sh = float(fy_sh)
    s = float(tightspacing)
    Mn = float(Mn)
    H = float(H)

    Ag = b*h  # 混凝土柱斷面積   
    I = 1/12*b*h**3
    E = 12000*fc**0.5
    K = 10*6*E*I/H

    # 避免除以零的情況
    if Ag != 0 and fc != 0 and H !=0 and s !=0 :
        Axialratio = P / (Ag * fc)  # 軸力比
        ρ = As/Ag*100 #主筋鋼筋比
        ρ_sh = A_sh/Ag*100 #剪力筋鋼筋比
        #-----------------------------------------------------------------------
        # 破壞形式計算(Vm Vn 比大小)
        #-----------------------------------------------------------------------

        Vc = 0.53*(1+P/(140*Ag))*(fc**(1/2))*b*d

        Sigma = P/Ag   

        ft = 1.06*(fc**(1/2))

        Alpha = (math.pi/4)-(1/2)*(math.atan(Sigma/(2*ft*((1+Sigma/ft)**(1/2))))) 

        Vs = (A_sh*fy_sh*d/s)/math.tan(Alpha)

        Vn = (Vc + Vs)/1000 #柱斷面之剪力計算強度 Vn (kgf除1000變tf)

        Vm = 2*Mn/H #雙曲率柱撓曲強度 

        if Vn > Vm:
            failuremode = 0 #撓曲,撓剪破壞
        elif Vn < Vm:
            failuremode = 1 #剪力破壞

    else:
        st.header("Please enter valid value")





    #-----------------------------------------------------------------------
    # 參數匯出
    #-----------------------------------------------------------------------

    # data = {'Column failure mode': failuremode,
    #         "Compressive strength of concrete f'c (kgf/cm^2)": fc,
    #         'Concrete cover c(cm)': c,
    #         'Cross-sectional area of the column(b*h)': Ag,
    #         'Height of the column':H ,
    #         'Yield stress of longitudinal reinforcement fy(kgf/cm)':fy ,
    #         'Yield stress of transverse reinforcement fy,sh(Mpa)':fy_sh ,
    #         'Cross-sectional area of the steel reinforcement':As ,
    #         'Area of shear reinforcement(cm2)': A_sh,
    #         'Ratio of total area of longitudinal reinforcement(As/Ag*100)':ρ,
    #         'ratio of total area of transverse reinforcement(Ash/Ag*100)':ρ_sh,
    #         'Tight spacing(cm)':tightspacing ,
    #         'Non-tight spacing(cm)': non_tightspacing,
    #         'axial load ratio':Axialratio ,
    #         'Hook angle':hook }

    # features = pd.DataFrame(data, index=[0])
    
    markdown_text = f'''
    ## Machine learning input features

    | Feature            | Value        | Reference values |
    | ------------------ | ------------ | ---------------- |
    | Column failure mode | {failuremode} | 0、1 |
    | Compressive strength of concrete f'c (kgf/cm^2) | {fc} | 112~1203 |
    | Concrete cover c(cm) | {c} | 0~6.51 |
    | Cross-sectional area of the column(b*h) | {Ag} | 64~4900 |
    | Height of the column | {H} | 8~330 |
    | Yield stress of longitudinal reinforcement fy(kgf/cm) | {fy} | 2800~7584 |
    | Yield stress of transverse reinforcement fy,sh(Mpa) | {fy_sh} | 2396~14520 |
    | Cross-sectional area of the steel reinforcement | {As} | 1.13~105.68 |
    | Area of shear reinforcement(cm2) | {A_sh} | 0.2~13.57 |
    | Ratio of total area of longitudinal reinforcement(As/Ag*100) | {ρ} | 0.68~6 |
    | ratio of total area of transverse reinforcement(Ash/Ag*100) | {ρ_sh} | 0.08~6.7 |
    | Tight spacing(cm) | {tightspacing} | 2~45 |
    | Non-tight spacing(cm) | {non_tightspacing} | 2~45 |
    | axial load ratio | {Axialratio} | 0.027~0.9 |
    | Hook angle | {hook} | 90、135、180 |
    '''

    st.markdown(markdown_text)
    




    with gzip.open('./model_pkl/randomForestModel_as.pgz', 'r') as f:
     randomForestModel_as = pickle.load(f)
    with gzip.open('./model_pkl/randomForestModel_c_S.pgz', 'r') as f:
     randomForestModel_c_S = pickle.load(f)
    with gzip.open('./model_pkl/randomForestModel_c_K.pgz', 'r') as f:
     randomForestModel_c_K = pickle.load(f)
    with gzip.open('./model_pkl/randomForestModel_c_A.pgz', 'r') as f:
     randomForestModel_c_A = pickle.load(f)
    with gzip.open('./model_pkl/randomForestModel_θp.pgz', 'r') as f:
     randomForestModel_θp = pickle.load(f)
    with gzip.open('./model_pkl/randomForestModel_c_C.pgz', 'r') as f:
     randomForestModel_c_C = pickle.load(f)
    with gzip.open('./model_pkl/randomForestModel_θpc.pgz', 'r') as f:
     randomForestModel_θpc = pickle.load(f)

    with gzip.open("model_pkl/z_score_scaler_X_as.pgz", "r") as f:
     z_score_scaler_X_as = pickle.load(f)
    with gzip.open("model_pkl/z_score_scaler_X_c_S.pgz", "r") as f:
     z_score_scaler_X_c_S = pickle.load(f)
    with gzip.open("model_pkl/z_score_scaler_X_c_K.pgz", "r") as f:
     z_score_scaler_X_c_K = pickle.load(f)
    with gzip.open("model_pkl/z_score_scaler_X_c_A.pgz", "r") as f:
     z_score_scaler_X_c_A = pickle.load(f)
    with gzip.open("model_pkl/z_score_scaler_X_θp.pgz", "r") as f:
     z_score_scaler_X_θp = pickle.load(f)
    with gzip.open("model_pkl/z_score_scaler_X_c_C.pgz", "r") as f:
     z_score_scaler_X_c_C = pickle.load(f)
    with gzip.open("model_pkl/z_score_scaler_X_θpc.pgz", "r") as f:
     z_score_scaler_X_θpc = pickle.load(f)


    # 使用PyGithub與GitHub API進行連動
    github_token = "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"  
    g = Github(github_token)
    repo = g.get_repo("Data-AI-Resilience-Group/ML-APP-RC-Plastic-Hinge")  

    # 設定下載路徑
    download_folder = "/mount/src/ML-APP-RC-Plastic-Hinge"

    # 下載Excel檔案
    app_file_path = "app輸入模型資料.xlsx" 
    content = repo.get_contents(app_file_path)
    downloaded_file_path = os.path.join(download_folder, "app輸入模型資料.xlsx")

    with open(downloaded_file_path, "wb") as f:
        f.write(content.decoded_content)


    # 下載Excel檔案
    user_file_path = "使用者匯入資料.xlsx"  # 替換為你的Excel檔案在GitHub中的路徑
    content = repo.get_contents(user_file_path)
    downloaded_file_path = os.path.join(download_folder, "使用者匯入資料.xlsx")

    with open(downloaded_file_path, "wb") as f:
        f.write(content.decoded_content)

    df_addNew = pd.read_excel("使用者匯入資料.xlsx", sheet_name="過往預測" )
    df_Pred = pd.read_excel("app輸入模型資料.xlsx", sheet_name="AI塑鉸預測" )

    #-----------------------------------------------------------------------
    ### 匯入使用者輸入資料至模型 ###
    #-----------------------------------------------------------------------
    
    df_Pred.iloc[:,[0]] = Specimen_Name  #試體名稱

    current_datetime = datetime.datetime.now()
    df_Pred.iloc[:, 1] = current_datetime  

    df_Pred.iloc[:,[3]] = fc

    df_Pred.iloc[:,[4]] = c

    df_Pred.iloc[:,[6]] = Ag

    df_Pred.iloc[:,[7]] = H

    df_Pred.iloc[:,[8]] = fy

    df_Pred.iloc[:,[12]] = fy_sh

    df_Pred.iloc[:,[9]] = As

    df_Pred.iloc[:,[14]] = A_sh

    df_Pred.iloc[:,[10]] = tightspacing

    df_Pred.iloc[:,[11]] = non_tightspacing

    df_Pred.iloc[:,[2]] = failuremode

    df_Pred.iloc[:,[5]] = Axialratio

    df_Pred.iloc[:,[13]] = hook

    #-----------------------------------------------------------------------
    #預測as
    #-----------------------------------------------------------------------

    Xdata_test_as= pd.DataFrame([[fc, c, Axialratio, Ag, H, fy, ρ, As, tightspacing, ρ_sh, failuremode, non_tightspacing]], 
             columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","箍筋面積比ρt(%)","破壞形式","非緊密間距(cm)"])

    # 使用標準化Z-Score套件
    # 對數據進行標準化 

    Xdata_test_Scale_as = z_score_scaler_X_as.transform(Xdata_test_as)

    # 使用測試資料預測  
    pred_as_ = randomForestModel_as.predict(Xdata_test_Scale_as).round(4)
    df_pred_as = pd.DataFrame(pred_as_, columns= ['as'])
        
    df_Pred.iloc[:,[15]]= df_pred_as  #寫入格式    
    #-----------------------------------------------------------------------
    #預測c_S
    #-----------------------------------------------------------------------

    Xdata_test_c_S = pd.DataFrame([[failuremode, fc, c, Axialratio, Ag, H, fy, ρ, tightspacing, fy_sh, A_sh, ρ_sh, non_tightspacing, pred_as_]], 
                              columns=["破壞形式", "混凝土強度f'c(kgf/cm^2)", "保護層(cm)", "軸力比", "斷面積(cm2)", "柱高(cm)", "降伏強度fy(kgf/cm)", "主筋面積比ρL(%)", "緊密間距(cm)", "fyt(MPa)", "Av(cm2)", "箍筋面積比ρt(%)", "非緊密間距(cm)", "as"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_S = z_score_scaler_X_c_S.transform(Xdata_test_c_S)

     
    # 使用測試資料預測 
    pred_c_S = randomForestModel_c_S.predict(Xdata_test_Scale_c_S).round(2)
    df_pred_c_S = pd.DataFrame(pred_c_S, columns= ['c_S'])
        
    df_Pred.iloc[:,[16]]= df_pred_c_S  #寫入格式    
    #-----------------------------------------------------------------------
    #預測c_K
    #-----------------------------------------------------------------------

    Xdata_test_c_K= pd.DataFrame([[fc, c, H, As, A_sh, ρ_sh, failuremode]], 
                     columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","柱高(cm)","鋼筋斷面積As","Av(cm2)","箍筋面積比ρt(%)","破壞形式"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_K = z_score_scaler_X_c_K.transform(Xdata_test_c_K)

    
    # 使用測試資料預測 
    pred_c_K = randomForestModel_c_K.predict(Xdata_test_Scale_c_K).round(4)
    df_pred_c_K = pd.DataFrame(pred_c_K, columns= ['c_K'])
        
    df_Pred.iloc[:,[19]]= df_pred_c_K  #寫入格式      
    #-----------------------------------------------------------------------
    #預測c_A
    #-----------------------------------------------------------------------

    Xdata_test_c_A= pd.DataFrame([[failuremode, fc, c, Axialratio, Ag, H, fy, ρ, As, tightspacing, fy_sh, hook,A_sh, ρ_sh, pred_as_, pred_c_K , non_tightspacing]], 
                     columns = ["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","as","c_K","非緊密間距(cm)"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_A = z_score_scaler_X_c_A.transform(Xdata_test_c_A)

    
    # 使用測試資料預測 
    pred_c_A = randomForestModel_c_A.predict(Xdata_test_Scale_c_A).round(4)
    df_pred_c_A = pd.DataFrame(pred_c_A, columns= ['c_A'])
        
    df_Pred.iloc[:,[18]]= df_pred_c_A  #寫入格式     
    #-----------------------------------------------------------------------
    #預測θp
    #-----------------------------------------------------------------------

    Xdata_test_θp= pd.DataFrame([[failuremode, fc, c, Axialratio, Ag, H, fy, ρ, As, tightspacing, non_tightspacing, fy_sh, hook, A_sh, ρ_sh, pred_c_S, pred_as_, pred_c_K ,pred_c_A ]], 
                     columns = ["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_K","c_A"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_θp = z_score_scaler_X_θp.transform(Xdata_test_θp)

    
    # 使用測試資料預測 
    pred_θp = randomForestModel_θp.predict(Xdata_test_Scale_θp).round(4)
    df_pred_θp = pd.DataFrame(pred_θp, columns= ['θp'])
        
    df_Pred.iloc[:,[20]]= df_pred_θp  #寫入格式    
    #-----------------------------------------------------------------------
    #預測c_C
    #-----------------------------------------------------------------------

    Xdata_test_c_C= pd.DataFrame([[fc, c, Axialratio, Ag, H, fy, ρ, As, tightspacing, fy_sh, A_sh, ρ_sh, pred_θp, failuremode, non_tightspacing]], 
                     columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","fyt(MPa)","Av(cm2)","箍筋面積比ρt(%)","θp","破壞形式","非緊密間距(cm)"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_c_C = z_score_scaler_X_c_C.transform(Xdata_test_c_C)

    
    # 使用測試資料預測 
    pred_c_C = randomForestModel_c_C.predict(Xdata_test_Scale_c_C).round(4)
    df_pred_c_C = pd.DataFrame(pred_c_C, columns= ['c_C'])
        
    df_Pred.iloc[:,[17]]= df_pred_c_C  #寫入格式      
    #-----------------------------------------------------------------------
    #預測_θpc
    #-----------------------------------------------------------------------

    Xdata_test_θpc= pd.DataFrame([[failuremode, fc, c, Axialratio, Ag, H, fy, ρ, As, tightspacing, non_tightspacing, fy_sh, hook, A_sh, ρ_sh, pred_c_S, pred_as_, pred_c_C, pred_c_K ,pred_c_A, pred_θp]], 
                     columns = ["破壞形式","混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","非緊密間距(cm)","fyt(MPa)","彎鉤角度","Av(cm2)","箍筋面積比ρt(%)","c_S","as","c_C","c_K","c_A","θp"])
    
    ## 使用標準化Z-Score套件
    ## 對數據進行標準化

    Xdata_test_Scale_θpc = z_score_scaler_X_θpc.transform(Xdata_test_θpc)

    
    # 使用測試資料預測 
    pred_θpc = randomForestModel_θpc.predict(Xdata_test_Scale_θpc).round(4)

    df_pred_θpc = pd.DataFrame(pred_θpc, columns= ['θpc'])
        
    df_Pred.iloc[:,[21]]= df_pred_θpc  #寫入格式      
    df_Pred.iloc[:,[22]]= K  #寫入格式  

    # 構建完整的檔案路徑
    output_file_path = os.path.join(download_folder, user_file_path)

    #輸出為EXCEL
    # df_Pred.to_excel('預測.xlsx',sheet_name='預測值',index=False)
    df_newData = pd.concat([df_addNew,df_Pred],axis = 0)
    df_newData.reset_index(drop = True ,inplace = True)
    df_newData.to_excel(output_file_path,sheet_name='過往預測',index=False)

    # 上傳更新後的Excel檔案
    with open(output_file_path, "rb") as f:
        data = f.read()
    repo.update_file("使用者匯入資料.xlsx", "Update data", data, content.sha, branch="main")

    # 刪除臨時檔案
    os.remove(os.path.join(download_folder, "使用者匯入資料.xlsx"))
    os.remove(os.path.join(download_folder, "app輸入模型資料.xlsx"))
    #-----------------------------------------------------------------------
    ### 預測完成，預測結果匯出 ###
    #-----------------------------------------------------------------------



    st.write('''       
    ---
    ## Prediction''')
    # Pred = {"a_s ":pred_as_,
    #               "c_S":pred_c_S,
    #               "c_C":pred_c_C,
    #               "c_A":pred_c_A,
    #               "c_K":pred_c_K,
    #               "θp":pred_θp,
    #               "θpc":pred_θpc
    #                }
    # Prediction = pd.DataFrame(Pred, index=[0]) 
    # st.write(Prediction) #顯示預測結果

    markdown_text = f'''

    | Output         | Value        |
    | ------------------ | ------------ |
    | Factor relating the concrete capacity to displacement ductility,K | {K} |
    | Strain hardening ratio,a_s | {pred_as_} |
    | Strength degradation rate under cyclic loading,c_S | {pred_c_S} |
    | Cyclic degradation rate after strength attenuation,c_C | {pred_c_C} |
    | Accelerated stiffness recovery degradation rate,c_A | {pred_c_A} |
    | Unloading stiffness degradation rate,c_K | {pred_c_K} |
    | Plastic rotation,θp | {pred_θp} |
    | Plastic rotation,θpc | {pred_θpc} |
    '''

    st.markdown(markdown_text)


    #-----------------------------------------------------------------------
    #開啟網頁終端機輸入:streamlit run Mainapp.py  
    #-----------------------------------------------------------------------
