import pandas as pd
from github import Github

# 使用PyGithub與GitHub API進行連動
g = Github("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")  # 替換為你的GitHub個人訪問令牌
repo = g.get_repo("Lipunpun/testapp")  # 替換為你的GitHub儲存庫路徑

# 下載Excel檔案


file_path = "輸入模型資料.xlsx"  # 替換為你的Excel檔案在GitHub中的路徑
content = repo.get_contents(file_path)
with open("輸入模型資料.xlsx", "wb") as f:
    f.write(content.decoded_content)

# 讀取 Excel 檔案並確認工作表名稱
xls = pd.ExcelFile("輸入模型資料.xlsx")
sheet_names = xls.sheet_names
print(sheet_names)

# 確認 'AI塑鉸預測' 工作表是否存在
if 'AI塑鉸預測' in sheet_names:
    # 讀取工作表 'AI塑鉸預測'
    df_Pred = pd.read_excel("輸入模型資料.xlsx", sheet_name="AI塑鉸預測", skiprows=1)
else:
    print("工作表 'AI塑鉸預測' 不存在。")

# 其他程式碼...
