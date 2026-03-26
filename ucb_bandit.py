import numpy as np
import matplotlib.pyplot as plt

def run_ucb_bandit(total_steps, c_values, true_means, num_runs=200):
    """
    執行多個 c (信心水準常數) 值域下的 UCB (Upper Confidence Bound) 演算法。
    """
    num_arms = len(true_means)
    
    results_mean_cum_avg = []
    results_std_cum_avg = []
    results_mean_total_reward = []
    
    for c in c_values:
        all_cumulative_averages = np.zeros((num_runs, total_steps))
        all_total_rewards = np.zeros(num_runs)
        
        for r in range(num_runs):
            q_values = np.zeros(num_arms)
            action_counts = np.zeros(num_arms)
            rewards_history = np.zeros(total_steps)
            
            # 初始化：每個機台都先拉一次，避免除以零
            for t in range(num_arms):
                action = t
                reward = np.random.normal(loc=true_means[action], scale=1.0)
                rewards_history[t] = reward
                
                action_counts[action] += 1
                q_values[action] = reward  # 直接賦值，因為是第一次
                
            # UCB 演算法迴圈
            for t in range(num_arms, total_steps):
                # t 代表目前所在的總步數 (從 0 開始算，所以到了第 t 步，其實前面已經過了 t 步)
                # UCB 公式: Q(a) + c * sqrt(ln(t) / N(a))
                ucb_values = q_values + c * np.sqrt(np.log(t) / action_counts)
                
                # 選取 UCB 值最大的動作 (處理平手)
                max_ucb = np.max(ucb_values)
                best_actions = np.where(np.isclose(ucb_values, max_ucb))[0]
                action = np.random.choice(best_actions)
                    
                # 觀察回報並加入雜訊
                reward = np.random.normal(loc=true_means[action], scale=1.0)
                rewards_history[t] = reward
                
                # 增量更新 (Incremental Implementation)
                action_counts[action] += 1
                q_values[action] += (reward - q_values[action]) / action_counts[action]
                
            all_cumulative_averages[r] = np.cumsum(rewards_history) / np.arange(1, total_steps + 1)
            all_total_rewards[r] = np.sum(rewards_history)
            
        results_mean_cum_avg.append(np.mean(all_cumulative_averages, axis=0))
        results_std_cum_avg.append(np.std(all_cumulative_averages, axis=0))
        results_mean_total_reward.append(np.mean(all_total_rewards))
        
    return results_mean_cum_avg, results_std_cum_avg, results_mean_total_reward

def main():
    # ---------- UCB 的探索常數 c 掃描 ----------
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    true_means = [0.8, 0.7, 0.5]
    total_steps = 10000
    num_runs = 200
    # ------------------------------------------------
    
    print(f"開始執行 UCB 模擬 ({num_runs} runs)...")
    mean_cum_avgs, std_cum_avgs, mean_tot_rewards = run_ucb_bandit(
        total_steps, c_values, true_means, num_runs)
    
    # ==========================
    # 繪製圖表
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    fig.suptitle(f"Upper Confidence Bound (UCB) Simulation | c sweep: {c_values}\nBandits: A={true_means[0]}, B={true_means[1]}, C={true_means[2]} | Budget: ${total_steps:,} | {num_runs} runs", 
                 fontsize=15, fontweight='bold')
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(c_values)))
    steps_array = np.arange(1, total_steps + 1)
    
    ax1.axhline(y=0.8, color='grey', linestyle=':', label='Optimal mean = 0.8')
    
    for i, c_val in enumerate(c_values):
        avg_line = mean_cum_avgs[i]
        std_line = std_cum_avgs[i]
        tot_rev = mean_tot_rewards[i]
        
        ax1.plot(steps_array, avg_line, label=f"c = {c_val} (total ≈ {int(tot_rev)})", color=colors[i], linewidth=2)
        ax1.fill_between(steps_array, avg_line - std_line, avg_line + std_line, color=colors[i], alpha=0.1)
        
    ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=13)
    ax1.set_xlabel("Dollars Spent", fontsize=11)
    ax1.set_ylabel("Average Return per Dollar", fontsize=11)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0.4, 0.95)
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    x_labels = [str(c) for c in c_values]
    ax2.axhline(y=8000, color='grey', linestyle=':', label='Theoretical max (8,000)')
    
    bars = ax2.bar(x_labels, mean_tot_rewards, color=colors, width=0.55)
    
    for bar, val in zip(bars, mean_tot_rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{int(val):,}", 
                 ha='center', va='bottom', fontweight='bold', color='black')
        
    ax2.set_title("Total Reward vs. Confidence Degree (c)", fontsize=13)
    ax2.set_xlabel("Confidence Degree (c)", fontsize=11)
    ax2.set_ylabel("Average Total Reward", fontsize=11)
    ax2.set_ylim(6000, 8400)
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filename = 'ucb_sweep.png'
    plt.savefig(filename, dpi=300)
    print(f"圖表已成功輸出並儲存至 '{filename}'")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    main()
