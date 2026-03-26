import numpy as np
import matplotlib.pyplot as plt

def run_ab_test_sweep(total_steps, exploration_budgets, true_means, num_runs=200):
    """
    透過不同的測試期預算長度（Test Budget），執行 A/B 測試模擬。
    與 Epsilon-Greedy 不同，A/B 測試的隨機過程可以在階段內完全向量化，因此執行速度極快。
    """
    num_arms = len(true_means)
    
    results_mean_cum_avg = []
    results_std_cum_avg = []
    results_mean_total_reward = []
    
    for explore_budget in exploration_budgets:
        all_cumulative_averages = np.zeros((num_runs, total_steps))
        all_total_rewards = np.zeros(num_runs)
        
        # 測試期必須平分給所有機台
        steps_per_arm = explore_budget // num_arms
        
        for r in range(num_runs):
            rewards_history = np.zeros(total_steps)
            
            # ==================================
            # 1. 測試期 (Test Phase, Pure Exploration)
            # ==================================
            if steps_per_arm > 0:
                empirical_means = np.zeros(num_arms)
                
                # 分別為每台機器產生測試期的回報 (向量化加速)
                for arm in range(num_arms):
                    start_idx = arm * steps_per_arm
                    end_idx = start_idx + steps_per_arm
                    
                    arm_rewards = np.random.normal(loc=true_means[arm], scale=1.0, size=steps_per_arm)
                    rewards_history[start_idx:end_idx] = arm_rewards
                    empirical_means[arm] = np.mean(arm_rewards)

                # 避開除不盡的問題
                actual_explore_budget = steps_per_arm * num_arms
                
                # 選出實際測試結果最好的機台（處理平手情形）
                max_mean = np.max(empirical_means)
                best_actions = np.where(np.isclose(empirical_means, max_mean))[0]
                winner_arm = np.random.choice(best_actions)
            else:
                # 如果測試預算為 0，等同於隨機盲猜一台到底
                actual_explore_budget = 0
                winner_arm = np.random.randint(num_arms)
                
            # ==================================
            # 2. 應用期 (Exploitation Phase)
            # ==================================
            remaining_steps = total_steps - actual_explore_budget
            if remaining_steps > 0:
                # 剩下的預算全部灌在勝出的那一台 (向量化加速)
                exploit_rewards = np.random.normal(loc=true_means[winner_arm], scale=1.0, size=remaining_steps)
                rewards_history[actual_explore_budget:] = exploit_rewards
                
            # 計算累積平均與總和
            all_cumulative_averages[r] = np.cumsum(rewards_history) / np.arange(1, total_steps + 1)
            all_total_rewards[r] = np.sum(rewards_history)
            
        results_mean_cum_avg.append(np.mean(all_cumulative_averages, axis=0))
        results_std_cum_avg.append(np.std(all_cumulative_averages, axis=0))
        results_mean_total_reward.append(np.mean(all_total_rewards))
        
    return results_mean_cum_avg, results_std_cum_avg, results_mean_total_reward

def main():
    # 這裡的掃描參數是「A/B 測試期總共分配多少預算」
    # 1500 是您最初設定的 (每台 500 次)
    # 9000 代表過度測試 (每台 3000 次，應用期只剩 1000 次)
    explore_budgets = [300, 900, 1500, 3000, 6000, 9000]
    true_means = [0.8, 0.7, 0.5]
    total_steps = 10000
    num_runs = 200
    
    print(f"開始執行傳統 A/B Test 參數掃描模擬 ({num_runs} runs)...")
    mean_cum_avgs, std_cum_avgs, mean_tot_rewards = run_ab_test_sweep(
        total_steps, explore_budgets, true_means, num_runs)
    
    # ==========================
    # 繪製圖表 (仿照 Epsilon-Greedy 套用的雙子圖格式)
    # ==========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 設定主標題
    fig.suptitle(f"Traditional A/B Test Simulation | Explore Budget sweep: {explore_budgets}\nBandits: A={true_means[0]}, B={true_means[1]}, C={true_means[2]} | Budget: ${total_steps:,} | {num_runs} runs", 
                 fontsize=15, fontweight='bold')
    
    # 取色帶
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(explore_budgets)))
    steps_array = np.arange(1, total_steps + 1)
    
    # --- 左圖：Cumulative Average Return vs. Dollars Spent ---
    ax1.axhline(y=0.8, color='grey', linestyle=':', label='Optimal mean = 0.8')
    
    for i, bgt in enumerate(explore_budgets):
        avg_line = mean_cum_avgs[i]
        std_line = std_cum_avgs[i]
        tot_rev = mean_tot_rewards[i]
        
        ax1.plot(steps_array, avg_line, label=f"Test Budget = {bgt} (total ≈ {int(tot_rev)})", color=colors[i], linewidth=2)
        ax1.fill_between(steps_array, avg_line - std_line, avg_line + std_line, color=colors[i], alpha=0.1)
        
    ax1.set_title("Cumulative Average Return vs. Dollars Spent", fontsize=13)
    ax1.set_xlabel("Dollars Spent", fontsize=11)
    ax1.set_ylabel("Average Return per Dollar", fontsize=11)
    ax1.set_xlim(0, total_steps)
    # 由於過少樣本的 A/B 測試選到 0.5 的機率很高，線條震盪幅度可能比 Epsilon-Greedy 大，可把原下界調低一點
    ax1.set_ylim(0.4, 0.95) 
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='lower right', fontsize=10)
    
    # --- 右圖：Total Reward vs. Test Budget Value ---
    x_labels = [str(b) for b in explore_budgets]
    ax2.axhline(y=8000, color='grey', linestyle=':', label='Theoretical max (8,000)')
    
    bars = ax2.bar(x_labels, mean_tot_rewards, color=colors, width=0.55)
    
    for bar, val in zip(bars, mean_tot_rewards):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{int(val):,}", 
                 ha='center', va='bottom', fontweight='bold', color='black')
        
    ax2.set_title("Total Reward vs. Test Phase Budget", fontsize=13)
    ax2.set_xlabel("Test Phase Budget (Total exploration steps)", fontsize=11)
    ax2.set_ylabel("Average Total Reward", fontsize=11)
    
    # y 下界適當拉低捕捉錯誤決策的下限
    ax2.set_ylim(6000, 8400) 
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    filename = 'ab_test_sweep.png'
    plt.savefig(filename, dpi=300)
    print(f"圖表已成功套用格式並儲存至 '{filename}'")
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    main()
