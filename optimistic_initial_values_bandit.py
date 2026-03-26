import numpy as np
import matplotlib.pyplot as plt

def run_oiv_sweep(total_steps, q0_values, true_means, alpha=0.1, num_runs=200):
    """
    執行 Optimistic Initial Values (樂觀初始值) 演算法的掃描。
    使用固定的 Step Size (alpha) 以確保初始值帶來的探索效果能穩定延續。
    """
    num_arms = len(true_means)
    
    results_mean_cum_avg = []
    results_std_cum_avg = []
    results_mean_total_reward = []
    
    for q0 in q0_values:
        all_cumulative_averages = np.zeros((num_runs, total_steps))
        all_total_rewards = np.zeros(num_runs)
        
        for r in range(num_runs):
            # 這裡就是 OIV 的精髓：Q 值不再從 0 開始，而是從一個樂觀的大數字 Q0 開始！
            q_values = np.full(num_arms, q0, dtype=float)
            rewards_history = np.zeros(total_steps)
            
            for t in range(total_steps):
                # 完全貪婪 (Epsilon = 0)，只根據當下 Q 值做決定
                max_q = np.max(q_values)
                best_actions = np.where(np.isclose(q_values, max_q))[0]
                action = np.random.choice(best_actions)
                
                # 取得回報
                reward = np.random.normal(loc=true_means[action], scale=1.0)
                rewards_history[t] = reward
                
                # 由於使用樂觀初始值，標準做法是搭配固定的學習率 (Constant Step Size)
                # 這樣才能讓初始值在不斷拉出低於期望的分數後「緩慢下降」，進而強迫演算法嘗試別的機台
                q_values[action] += alpha * (reward - q_values[action])
                
            all_cumulative_averages[r] = np.cumsum(rewards_history) / np.arange(1, total_steps + 1)
            all_total_rewards[r] = np.sum(rewards_history)
            
        results_mean_cum_avg.append(np.mean(all_cumulative_averages, axis=0))
        results_std_cum_avg.append(np.std(all_cumulative_averages, axis=0))
        results_mean_total_reward.append(np.mean(all_total_rewards))
        
    return results_mean_cum_avg, results_std_cum_avg, results_mean_total_reward


def main():
    # 掃描不同的初始樂觀估計值 (Q0)
    # 我們設定真實機台的回報最大只有 0.8
    # 0.0 代表毫無樂觀 (在 epsilon=0 下極易陷入局部最佳)
    # 1.0, 2.0 代表輕度樂觀
    # 5.0, 10.0 代表過度樂觀 (要花很久時間才會對機台失望)
    q0_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    true_means = [0.8, 0.7, 0.5]
    total_steps = 10000
    num_runs = 200
    alpha = 0.1
    
    print(f"開始執行 Optimistic Initial Values 模擬 ({num_runs} runs)... 等待幾秒鐘...")
    mean_cum_avgs, std_cum_avgs, mean_tot_rewards = run_oiv_sweep(
        total_steps, q0_values, true_means, alpha, num_runs)
    
    # ==========================
    # 雙子圖視覺化輸出
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    fig.suptitle(f"Optimistic Initial Values Simulation | Q0 sweep: {q0_values}\nBandits: A={true_means[0]}, B={true_means[1]}, C={true_means[2]} | α={alpha} | Budget: ${total_steps:,} | {num_runs} runs", 
                 fontsize=15, fontweight='bold')
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(q0_values)))
    steps_array = np.arange(1, total_steps + 1)
    
    # --- 左圖：累積平均回報 ---
    ax1.axhline(y=0.8, color='grey', linestyle=':', label='Optimal mean = 0.8')
    
    for i, q0 in enumerate(q0_values):
        avg_line = mean_cum_avgs[i]
        std_line = std_cum_avgs[i]
        tot_rev = mean_tot_rewards[i]
        
        ax1.plot(steps_array, avg_line, label=f"Q0 = {q0} (total ≈ {int(tot_rev)})", color=colors[i], linewidth=2)
        ax1.fill_between(steps_array, avg_line - std_line, avg_line + std_line, color=colors[i], alpha=0.1)
        
    ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=13)
    ax1.set_xlabel("Dollars Spent", fontsize=11)
    ax1.set_ylabel("Average Return per Dollar", fontsize=11)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0.4, 0.95)
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='lower left', fontsize=10)
    
    # --- 右圖：總回報長條圖 ---
    x_labels = [str(q) for q in q0_values]
    ax2.axhline(y=8000, color='grey', linestyle=':', label='Theoretical max (8,000)')
    
    bars = ax2.bar(x_labels, mean_tot_rewards, color=colors, width=0.55)
    
    for bar, val in zip(bars, mean_tot_rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{int(val):,}", 
                 ha='center', va='bottom', fontweight='bold', color='black')
        
    ax2.set_title("Total Reward vs. Initial Q-Value (Q0)", fontsize=13)
    ax2.set_xlabel("Optimistic Initial Value (Q0)", fontsize=11)
    ax2.set_ylabel("Average Total Reward", fontsize=11)
    ax2.set_ylim(4000, 8400) # 若太過樂觀，總分可能會更低
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filename = 'optimistic_initial_values_sweep.png'
    plt.savefig(filename, dpi=300)
    print(f"圖表已成功輸出並儲存至 '{filename}'")
    plt.close() # 執行後無需畫出，直接存檔給後續說明使用即可
    # plt.show() 

if __name__ == "__main__":
    np.random.seed(42)  # 確保結果可重播
    main()
