# 強化學習：多臂賭博機 (Multi-Armed Bandit) 經典演算法實作與比較

這是一個深入探討強化學習中經典「多臂賭博機（Multi-Armed Bandit, MAB）」問題的 Python 實作專案。我們透過 10,000 步的預算模擬，比較了在未知的回報分佈下，不同演算法如何平衡「探索 (Exploration)」與「利用 (Exploitation)」，並找出最大化整體收益的最佳解。

## ⚙️ 模擬環境設定

- **選項 (Arms)**：共 3 台老虎機（A, B, C）
- **真實期望回報**：分別為 0.8, 0.7, 0.5
- **回報雜訊分佈**：每一拉桿獲得的回報服從常態分佈，標準差設為 1.0
- **預算限制**：總共 10,000 步
- **統計穩定度**：為了降低雜訊讓圖表更加平滑易讀，所有的對比都會進行高達 200 次獨立實驗取平均與標準差。

## 🚀 實作演算法清單

本專案共探討並實作了六種不同的策略：

### 1. 傳統 A/B 測試 (A/B Testing)
- **程式碼**：[`ab_test_bandit.py`](./ab_test_bandit.py)
- **詳情**：[`AB_Test_Introduction.md`](./AB_Test_Introduction.md)
- **核心概念**：強行切分測試期（純探索）與應用期（純利用），測試期結束後直接賭在當時表現最好的機台上直至結束。

### 2. Epsilon-Greedy ($\epsilon$-Greedy)
- **程式碼**：[`epsilon_greedy_bandit.py`](./epsilon_greedy_bandit.py)
- **詳情**：[`Epsilon_Greedy_Introduction.md`](./Epsilon_Greedy_Introduction.md)
- **核心概念**：保留 $\epsilon$ 的機率進行盲目隨機探索，其餘 $(1-\epsilon)$ 的機率選擇目前平均回報最高的機台。

### 3. 樂觀初始值 (Optimistic Initial Values)
- **程式碼**：[`optimistic_initial_values_bandit.py`](./optimistic_initial_values_bandit.py)
- **詳情**：[`Optimistic_Initial_Values_Introduction.md`](./Optimistic_Initial_Values_Introduction.md)
- **核心概念**：在最初給予所有機台極高且不切實際的初始期望分數，使得只要機器一被拉動，分數掉下來就會迫使我們去拉其他期待值還很高的機器，達到自然而然輪流探索的效果。

### 4. Softmax (Boltzmann) 探索
- **程式碼**：[`softmax_bandit.py`](./softmax_bandit.py)
- **詳情**：[`Softmax_Introduction.md`](./Softmax_Introduction.md)
- **核心概念**：根據老虎機目前分數（Q 值）的比例決定探索機率。爛機台會遭到幾乎 100% 機率的抹除，而次佳機能保有微小的被測試機會，比起 Epsilon-Greedy 能帶來更平穩的後期發展。

### 5. Upper Confidence Bound (UCB) 
- **程式碼**：[`ucb_bandit.py`](./ucb_bandit.py)
- **詳情**：[`UCB_Introduction.md`](./UCB_Introduction.md)
- **核心概念**：運用數學統計算出每一個機台平均分數加上其「不確定性上界」，將所有的資源完美地安排在最高潛力的未探索項目或是目前最佳解上，完全消除隨機盲猜。

### 6. Thompson Sampling (貝氏推論)
- **程式碼**：[`thompson_sampling_bandit.py`](./thompson_sampling_bandit.py)
- **詳情**：[`Thompson_Sampling_Introduction.md`](./Thompson_Sampling_Introduction.md)
- **核心概念**：為每個老虎機設定一個回報的先驗機率分佈模型，並於每回合從分佈中「抽樣」；根據真實收集到的回報去更新後驗分佈。能用極短時間將次佳解排掉並幾乎完全聚焦在完美最佳解上。

## 📊 視覺化圖表比較

執行任一演算法的 Python 檔案，皆會產生對應的 `_sweep.png` 圖表。圖表一律包含：
1. **累積平均回報 (Cumulative Average Return)** 曲線，以及標準差陰影區間。
2. **總體回報柱狀圖 (Total Reward vs 超參數)**，呈現參數調整對最終總資產所造成的影響。

---

> 👨‍💻 本專案所有實作皆採用 Numpy 進行效能與向量化優化操作。
