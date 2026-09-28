import pandas as pd
import sys

if len(sys.argv) < 2:
    print("请指定 CSV 文件路径，例如: python fix_csv.py ./data/au.csv")
    sys.exit(1)
file_path = sys.argv[1]
try:
    print(f"正在处理: {file_path} ...")
    df = pd.read_csv(file_path)
    print("修改前列名：")
    print(df.columns.tolist())

    new_columns = []
    for col in df.columns:
        if col.startswith("AU"):
            new_columns.append(" " + col)
        elif col.startswith(" AU"):
            new_columns.append(col)
        else:
            new_columns.append(col)
    df.columns = new_columns
    df.to_csv(file_path, index=False)

    print("\n✅ 修复完成！")
    print("修改后列名：")
    print(df.columns.tolist())

except Exception as e:
    print(f"❌ 发生错误: {e}")
