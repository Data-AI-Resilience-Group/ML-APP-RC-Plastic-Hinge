import pandas as pd
from github import Github
import pickle
import gzip

with gzip.open('./model_pkl/randomForestModel_as.pgz', 'r') as f:
    randomForestModel_as = pickle.load(f)

with gzip.open("model_pkl/z_score_scaler_X_as.pgz", "r") as f:
    z_score_scaler_X_as = pickle.load(f)


#-----------------------------------------------------------------------
#預測as
#-----------------------------------------------------------------------
fc=280
concretecover=6.35 
Axialratio=0.017986
Ag=3600
columnheight=427
fy=4696
As=108.36
ρ=3
tightspacing=10
ρ_sh=0.31
failuremode=1
widespacing=10

Xdata_test_as= pd.DataFrame([[fc, concretecover, Axialratio, Ag, columnheight, fy, ρ, As, tightspacing, ρ_sh, failuremode, widespacing]], 
            columns = ["混凝土強度f'c(kgf/cm^2)","保護層(cm)","軸力比","斷面積(cm2)","柱高(cm)","降伏強度fy(kgf/cm)","主筋面積比ρL(%)","鋼筋斷面積As","緊密間距(cm)","箍筋面積比ρt(%)","破壞形式","非緊密間距(cm)"])

# 使用標準化Z-Score套件
# 對數據進行標準化 

Xdata_test_Scale_as = z_score_scaler_X_as.transform(Xdata_test_as)

# 使用測試資料預測  
pred_as_ = randomForestModel_as.predict(Xdata_test_Scale_as).round(4)
print(pred_as_)
    
