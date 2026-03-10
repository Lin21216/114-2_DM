# 第 4 週作業：簡單線性迴歸與多元線性迴歸

## 作業資訊

| 項目 | 說明 |
|------|------|
| 對應教科書 | Ch4 簡單線性迴歸、Ch5 多元線性迴歸 |
| 繳交方式 | 在 Fork 的 week04/ 資料夾中建立三個檔案，發 PR 繳交 |
| 繳交期限 | 下週上課前 |
| PR 標題格式 | 學號_姓名_week04 |

---

## 第 1 題：簡單線性迴歸建模與視覺化（40 分）

### 任務說明

使用 Scikit-learn 的 LinearRegression，對一份海洋觀測資料進行簡單線性迴歸建模。你需要完成資料探索、模型訓練、預測評估與視覺化。

### 測試資料

請使用以下程式碼建立測試資料：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50

# 模擬海洋觀測資料：風速與波浪高度的關係
wind_speed = np.random.uniform(5, 35, n)
wave_height = 0.08 * wind_speed + 0.3 + np.random.normal(0, 0.3, n)

df = pd.DataFrame({
    'wind_speed': np.round(wind_speed, 1),
    'wave_height': np.round(wave_height, 2)
})
df.to_csv('ocean_obs.csv', index=False)
print("資料前 5 筆：")
print(df.head())
print(f"\n資料形狀：{df.shape}")
print(f"\n基本統計：\n{df.describe()}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 讀取 ocean_obs.csv，繪製 wind_speed 與 wave_height 的散布圖
2. 計算兩個變數的相關係數
3. 將資料切割為訓練集和測試集（test_size=0.2, random_state=42）
4. 使用 LinearRegression 訓練模型
5. 印出迴歸係數和截距
6. 計算訓練集與測試集的 R² 分數
7. 繪製散布圖加上迴歸線

### 作答內容

請建立 `week04/q1_simple_regression.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 資料探索 ===
（貼上前 5 筆資料、describe 統計、相關係數）

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 模型結果 ===
迴歸係數（斜率）：???
截距：???
訓練集 R² 分數：???
測試集 R² 分數：???

=== 散布圖與迴歸線 ===
（貼上圖片或描述圖表呈現的結果）
```

### 提示

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = df[['wind_speed']]
y = df['wave_height']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(f"斜率：{model.coef_[0]:.4f}")
print(f"截距：{model.intercept_:.4f}")
print(f"訓練集 R²：{model.score(X_train, y_train):.4f}")
print(f"測試集 R²：{model.score(X_test, y_test):.4f}")
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 資料探索完成：散布圖、相關係數 | 10 分 |
| 正確切割資料並訓練模型 | 10 分 |
| 印出迴歸係數、截距、R² 分數 | 10 分 |
| 繪製散布圖加上迴歸線 | 10 分 |

---

## 第 2 題：多元迴歸與 Pipeline 特徵比較（40 分）

### 任務說明

使用多元線性迴歸搭配 Pipeline，比較不同特徵組合對預測效能的影響。加入 PolynomialFeatures 觀察多次方迴歸是否能改善預測。

### 測試資料

請使用以下程式碼建立測試資料：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

# 模擬港口貨物量預測資料
data = {
    'berth_count': np.random.randint(5, 30, n),
    'crane_count': np.random.randint(2, 15, n),
    'channel_depth': np.round(np.random.uniform(8, 18, n), 1),
    'storage_area': np.random.randint(50, 500, n),
    'distance_to_city': np.round(np.random.uniform(1, 50, n), 1)
}
df = pd.DataFrame(data)

df['cargo_volume'] = (
    1200 * df['berth_count'] +
    3500 * df['crane_count'] +
    800 * df['channel_depth'] +
    50 * df['storage_area'] -
    200 * df['distance_to_city'] +
    np.random.normal(0, 5000, n)
).astype(int)

df.to_csv('port_cargo.csv', index=False)
print("資料前 5 筆：")
print(df.head())
print(f"\n相關係數：\n{df.corr()['cargo_volume'].sort_values(ascending=False)}")
```

### Python 程式要求

撰寫程式碼完成以下工作：

1. 讀取資料，計算各特徵與 cargo_volume 的相關係數
2. 切割資料（test_size=0.2, random_state=42）
3. 建立 Pipeline 1：StandardScaler → LinearRegression（使用全部特徵）
4. 建立 Pipeline 2：StandardScaler → LinearRegression（只使用相關係數前 3 高的特徵）
5. 建立 Pipeline 3：StandardScaler → PolynomialFeatures(degree=2) → LinearRegression（使用相關係數最高的 2 個特徵）
6. 分別計算三個 Pipeline 的訓練集與測試集 R² 分數
7. 比較三種方式的結果，印出比較表

### 作答內容

請建立 `week04/q2_multiple_regression.txt`，依照以下格式填寫：

```
姓名：
學號：

=== 各特徵與 cargo_volume 的相關係數 ===
（貼上相關係數排序結果）

=== 完整程式碼 ===
（貼上你撰寫的完整 Python 程式碼）

=== 三種 Pipeline 的 R² 分數比較 ===
Pipeline 1（全部特徵）：訓練 R²=???  測試 R²=???
Pipeline 2（前 3 高特徵）：訓練 R²=???  測試 R²=???
Pipeline 3（前 2 特徵 + 多次方）：訓練 R²=???  測試 R²=???

=== 觀察與發現 ===
（用 1-2 句話說明哪種 Pipeline 效果最好，你認為原因是什麼）
```

### 提示

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('reg', LinearRegression())
])
pipe.fit(X_train, y_train)
print(f"R²：{pipe.score(X_test, y_test):.4f}")
```

### 評分標準

| 項目 | 配分 |
|------|------|
| 正確計算相關係數並選擇特徵 | 8 分 |
| Pipeline 1 建立正確且有結果 | 8 分 |
| Pipeline 2 建立正確且有結果 | 8 分 |
| Pipeline 3 使用 PolynomialFeatures 且有結果 | 8 分 |
| 比較表完整，觀察合理 | 8 分 |

---

## 第 3 題：迴歸觀念題（20 分）

### 作答內容

請建立 `week04/q3_concept.txt`，回答以下問題：

```
姓名：
學號：

Q1：R² 分數的意義是什麼？如果 R² = 0.85，代表什麼？
    如果測試集的 R² 遠低於訓練集的 R²，可能代表什麼問題？
A1：???

Q2：PolynomialFeatures(degree=2) 會對原始特徵做什麼轉換？
    舉例說明：如果原始特徵有 x1 和 x2 兩個，degree=2 會產生哪些新特徵？
    在什麼情境下，多次方迴歸會比線性迴歸效果好？
A2：???

Q3：在多元迴歸中，為什麼不是特徵越多效果越好？
    請說明加入不相關特徵可能造成的問題。
A3：???
```

### 評分標準

| 項目 | 配分 |
|------|------|
| Q1 正確解釋 R² 意義，說明訓練/測試 R² 差距的問題 | 7 分 |
| Q2 正確說明 PolynomialFeatures 轉換，舉例正確 | 7 分 |
| Q3 合理說明特徵過多的問題 | 6 分 |

---

## 繳交 Checklist

- [ ] week04/q1_simple_regression.txt 包含完整程式碼、模型結果與散布圖
- [ ] week04/q2_multiple_regression.txt 包含完整程式碼與三種 Pipeline 比較
- [ ] week04/q3_concept.txt 包含三題觀念回答
- [ ] 已 push 到自己的 Fork
- [ ] 已發 PR，標題格式：學號_姓名_week04
