import pandas as pd

# 複数のURLをリストとして指定
urls = [
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0017.html',
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0018.html',
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0019.html',
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0020.html',
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0021.html',
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0022.html',
    'https://www.city.kitakyushu.lg.jp/kankyou/file_0023.html',
]

all_dfs = []  # 全テーブルを保存するリスト

# 各URLに対してテーブルを読み込み
for url in urls:
    try:
        # URLから全てのテーブルを読み込む
        dfs = pd.read_html(url)
        all_dfs.extend(dfs)  # リストにデータフレームを追加
    except Exception as e:
        print(f"Failed to process {url}. Error: {e}")
        print("\n")

# 全てのデータフレームを一つに連結
combined_df = pd.concat(all_dfs, ignore_index=True)

# CSVファイルとして出力
combined_df.to_csv('./combined_tables_c.csv', index=False, encoding='utf-8')
print("All tables have been saved to combined_tables.csv")
