import numpy as np
import matplotlib.pyplot as plt

def run_thompson_sampling_bandit(total_steps, prior_stds, true_means, num_runs=200):
    """
    執行常態分佈回報情況下的 Thompson Sampling 演算法，對不同的先驗標準差 (Prior Std) 進行掃描。
    假設回報分佈已知為常態分佈且標準差 (sigma) = 1.0，先驗平均數預設為 0.0。
    """
    num_arms = len(true_means)
    
    results_mean_cum_avg = []
    results_std_cum_avg = []
    results_mean_total_reward = []
    
    # 實際回報的變異數為 1.0^2 = 1.0
    reward_var = 1.0
    prior_mean = 0.0
    
    for prior_std in prior_stds:
        all_cumulative_averages = np.zeros((num_runs, total_steps))
        all_total_rewards = np.zeros(num_runs)
        
        # 預先計算先驗精確度 (Prior Precision)
        prior_precision = 1.0 / (prior_std ** 2)
        
        for r in range(num_runs):
            action_counts = np.zeros(num_arms)
            sum_rewards = np.zeros(num_arms)
            rewards_history = np.zeros(total_steps)
            
            for t in range(total_steps):
                # Thompson Sampling 動作選擇：針對每個機台從其後驗分佈中抽樣
                # 後驗變異數 = 1 / (先驗精確度 + 樣本數 / 實際變異數)
                # 後驗平均數 = 後驗變異數 * (先驗平均數 * 先驗精確度 + 樣本總和 / 實際變異數)
                posterior_vars = 1.0 / (prior_precision + action_counts / reward_var)
                posterior_means = posterior_vars * (prior_mean * prior_precision + sum_rewards / reward_var)
                
                # 從後驗分佈中抽取樣本 (Theta)
                sampled_thetas = np.random.normal(loc=posterior_means, scale=np.sqrt(posterior_vars))
                
                # 選擇被抽出最大值的機台 (處理平手)
                max_theta = np.max(sampled_thetas)
                best_actions = np.where(np.isclose(sampled_thetas, max_theta))[0]
                action = np.random.choice(best_actions)
                    
                # 觀察回報並加入雜訊
                reward = np.random.normal(loc=true_means[action], scale=np.sqrt(reward_var))
                rewards_history[t] = reward
                
                # 更新狀態 (Incremental Implementation)
                action_counts[action] += 1
                sum_rewards[action] += reward
                
            all_cumulative_averages[r] = np.cumsum(rewards_history) / np.arange(1, total_steps + 1)
            all_total_rewards[r] = np.sum(rewards_history)
            
        results_mean_cum_avg.append(np.mean(all_cumulative_averages, axis=0))
        results_std_cum_avg.append(np.std(all_cumulative_averages, axis=0))
        results_mean_total_reward.append(np.mean(all_total_rewards))
        
    return results_mean_cum_avg, results_std_cum_avg, results_mean_total_reward

def main():
    # ---------- Thompson Sampling 的先驗標準差 (Prior Std) 掃描 ----------
    prior_stds = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    true_means = [0.8, 0.7, 0.5]
    total_steps = 10000
    num_runs = 200
    # ---------------------------------------------------------------------
    
    print(f"開始執行 Thompson Sampling 模擬 ({num_runs} runs)...")
    mean_cum_avgs, std_cum_avgs, mean_tot_rewards = run_thompson_sampling_bandit(
        total_steps, prior_stds, true_means, num_runs)
    
    # ==========================
    # 繪製圖表
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    fig.suptitle(f"Thompson Sampling Simulation | Prior Std ($\sigma_0$) sweep: {prior_stds}\nBandits: A={true_means[0]}, B={true_means[1]}, C={true_means[2]} | Budget: ${total_steps:,} | {num_runs} runs", 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(prior_stds)))
    steps_array = np.arange(1, total_steps + 1)
    
    ax1.axhline(y=0.8, color='grey', linestyle=':', label='Optimal mean = 0.8')
    
    for i, p_std in enumerate(prior_stds):
        avg_line = mean_cum_avgs[i]
        std_line = std_cum_avgs[i]
        tot_rev = mean_tot_rewards[i]
        
        ax1.plot(steps_array, avg_line, label=f"$\sigma_0$ = {p_std} (total ≈ {int(tot_rev)})", color=colors[i], linewidth=2)
        ax1.fill_between(steps_array, avg_line - std_line, avg_line + std_line, color=colors[i], alpha=0.1)
        
    ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=13)
    ax1.set_xlabel("Dollars Spent", fontsize=11)
    ax1.set_ylabel("Average Return per Dollar", fontsize=11)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0.4, 0.95)
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    x_labels = [str(p) for p in prior_stds]
    ax2.axhline(y=8000, color='grey', linestyle=':', label='Theoretical max (8,000)')
    
    bars = ax2.bar(x_labels, mean_tot_rewards, color=colors, width=0.55)
    
    for bar, val in zip(bars, mean_tot_rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{int(val):,}", 
                 ha='center', va='bottom', fontweight='bold', color='black')
        
    ax2.set_title("Total Reward vs. Prior Standard Deviation ($\sigma_0$)", fontsize=13)
    ax2.set_xlabel("Prior Standard Deviation ($\sigma_0$)", fontsize=11)
    ax2.set_ylabel("Average Total Reward", fontsize=11)
    ax2.set_ylim(6000, 8400)
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filename = 'thompson_sampling_sweep.png'
    plt.savefig(filename, dpi=300)
    print(f"圖表已成功輸出並儲存至 '{filename}'")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    main()
