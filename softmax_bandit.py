import numpy as np
import matplotlib.pyplot as plt

def run_softmax_bandit(total_steps, taus, true_means, num_runs=200):
    """
    執行多個 Temperature (tau) 值域下的 Softmax (Boltzmann) 演算法。
    透過 num_runs 次的重複實驗來取得平均值與標準差，使圖表更穩定且具統計意義。
    """
    num_arms = len(true_means)
    
    results_mean_cum_avg = []
    results_std_cum_avg = []
    results_mean_total_reward = []
    
    for tau in taus:
        all_cumulative_averages = np.zeros((num_runs, total_steps))
        all_total_rewards = np.zeros(num_runs)
        
        for r in range(num_runs):
            q_values = np.zeros(num_arms)
            action_counts = np.zeros(num_arms)
            rewards_history = np.zeros(total_steps)
            
            for t in range(total_steps):
                # Softmax (Boltzmann) 動作選擇
                # 處理數值穩定性：為了避免 np.exp 遇到極大數值產生溢位，將所有 z 減去當前最大值
                z = q_values / tau
                z = z - np.max(z) 
                exp_z = np.exp(z)
                probs = exp_z / np.sum(exp_z)
                
                # 根據計算出的機率分佈選擇動作
                action = np.random.choice(num_arms, p=probs)
                    
                # 觀察回報並加入雜訊 (標準差 1.0 的常態分佈)
                reward = np.random.normal(loc=true_means[action], scale=1.0)
                rewards_history[t] = reward
                
                # 增量更新 Q 值 (Incremental Implementation)
                action_counts[action] += 1
                q_values[action] += (reward - q_values[action]) / action_counts[action]
                
            # 計算該 run 的累積平均回報與總金額回報
            all_cumulative_averages[r] = np.cumsum(rewards_history) / np.arange(1, total_steps + 1)
            all_total_rewards[r] = np.sum(rewards_history)
            
        # 計算該 tau 下 200 次 run 的「跨實驗平均」與「跨實驗標準差」
        results_mean_cum_avg.append(np.mean(all_cumulative_averages, axis=0))
        results_std_cum_avg.append(np.std(all_cumulative_averages, axis=0))
        results_mean_total_reward.append(np.mean(all_total_rewards))
        
    return results_mean_cum_avg, results_std_cum_avg, results_mean_total_reward

def main():
    # ---------- 參數設定與您提供的圖片一致 ----------
    taus = [0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
    true_means = [0.8, 0.7, 0.5]
    total_steps = 10000
    num_runs = 200
    # ------------------------------------------------
    
    print(f"開始執行 Softmax (Boltzmann) 模擬 ({num_runs} runs)... 這可能需要幾秒鐘。")
    mean_cum_avgs, std_cum_avgs, mean_tot_rewards = run_softmax_bandit(
        total_steps, taus, true_means, num_runs)
    
    # ==========================
    # 繪製雙子圖表 (完全仿照提供的圖片樣式)
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 設定主標題 (Super Title)
    fig.suptitle(f"Softmax (Boltzmann) Bandit Simulation | Temperature τ sweep: {taus}\nBandits: A={true_means[0]}, B={true_means[1]}, C={true_means[2]} | Budget: ${total_steps:,} | {num_runs} runs", 
                 fontsize=15, fontweight='bold')
    
    # 建立 Color Map (從亮藍色過渡到深紅色)，取 6 個不同的漸層顏色
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(taus)))
    
    # === 左圖：Cumulative Average Return vs. Dollars Spent ===
    steps_array = np.arange(1, total_steps + 1)
    
    # 畫上一條「理論最優平均線」 (0.8)
    ax1.axhline(y=0.8, color='grey', linestyle=':', label='Optimal mean = 0.8')
    
    for i, tau in enumerate(taus):
        avg_line = mean_cum_avgs[i]
        std_line = std_cum_avgs[i]
        tot_rev = mean_tot_rewards[i]
        
        # 繪製主平均線
        ax1.plot(steps_array, avg_line, label=f"τ = {tau} (total ≈ {int(tot_rev)})", color=colors[i], linewidth=2)
        # 繪製誤差陰影區間 (以 1 個標準差示意)
        ax1.fill_between(steps_array, avg_line - std_line, avg_line + std_line, color=colors[i], alpha=0.1)
        
    ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=13)
    ax1.set_xlabel("Dollars Spent", fontsize=11)
    ax1.set_ylabel("Average Return per Dollar", fontsize=11)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0.3, 0.95)
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    # === 右圖：Total Reward vs. Temperature Value ===
    x_labels = [str(t) for t in taus]
    
    # 畫上一條「理想最高總回報的虛線」 (8000)
    ax2.axhline(y=8000, color='grey', linestyle=':', label='Theoretical max (8,000)')
    
    bars = ax2.bar(x_labels, mean_tot_rewards, color=colors, width=0.55)
    
    # 在每根長條的最上方標記整數數值
    for bar, val in zip(bars, mean_tot_rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{int(val):,}", 
                 ha='center', va='bottom', fontweight='bold', color='black')
        
    ax2.set_title("Total Reward vs. Temperature (τ)", fontsize=13)
    ax2.set_xlabel("Temperature (τ)", fontsize=11)
    ax2.set_ylabel("Average Total Reward", fontsize=11)
    ax2.set_ylim(6000, 8400) # 適度調整邊界
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # 微調排版避免子圖與大標題重疊
    
    filename = 'softmax_sweep.png'
    plt.savefig(filename, dpi=300)
    print(f"圖表已成功輸出並儲存至 '{filename}'")
    
    # 建議不呼叫 plt.show() 以防在自動批次執行時卡住 (或是可保留以相容互動模式)
    plt.show()

if __name__ == "__main__":
    # 設定亂數種子，使得模擬軌跡具有較高的穩定性與可重現性
    np.random.seed(42)
    main()
