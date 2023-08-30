
from github import Github
import os
import pandas as pd

# 使用PyGithub與GitHub API進行連動
github_token = "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"  
g = Github(github_token)
repo = g.get_repo("Lipunpun/testapp")  

# 設定下載路徑
download_folder = "C:\\Users\\user\\Desktop\\新增資料夾"

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

a=500
df_Pred.iloc[:,[0]] = a  #試體名稱

# 儲存修改後的 DataFrame
df_Pred.to_excel("app輸入模型資料.xlsx", sheet_name="AI塑鉸預測", index=False)

# # 刪除臨時檔案
# os.remove(os.path.join(download_folder, "使用者匯入資料.xlsx"))
# os.remove(os.path.join(download_folder, "app輸入模型資料.xlsx"))
