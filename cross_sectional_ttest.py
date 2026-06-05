import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import os
import zipfile
import shutil

# Google Colab環境でのダウンロード用ライブラリ
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 1. 拡張シミュレーション基盤モデル (変更なし)
# ==========================================

def generate_asymmetric_trust_network(N, m):
    G = nx.barabasi_albert_graph(N, m)
    adj_matrix = nx.to_numpy_array(G)
    D_matrix = np.random.uniform(-1, 1, (N, N))
    D_matrix = D_matrix * adj_matrix
    return G, D_matrix

def cutoff_function(I_i, I_j, threshold):
    diff = np.abs(I_i - I_j)
    return 1.0 / (1.0 + np.exp(diff - threshold))

def update_opinions_stochastic(I, D, active_agents, A_base, alpha, c_array, noise_std, bubble_factor, threshold, dt=0.1):
    N = len(I)
    delta_I = np.zeros(N)
    active_indices = np.where(active_agents)[0]
    
    for i in active_indices:
        term1 = -alpha * I[i]
        
        A_effective = A_base if I[i] >= 0 else A_base * bubble_factor
        term2 = c_array[i] * A_effective
        
        term3 = 0
        for j in active_indices:
            if D[i, j] != 0:
                phi = cutoff_function(I[i], I[j], threshold)
                term3 += D[i, j] * phi * (I[j] - I[i])
        
        noise = np.random.normal(0, noise_std) * np.sqrt(dt)
        delta_I[i] = (term1 + term2 + term3) * dt + noise
        
    return I + delta_I

# ==========================================
# 2. 独立試行による断面データ抽出関数
# ==========================================

def run_single_trial_final_state(steps=150, N=300, m=3, noise_std=3.0, bubble_factor=0.1, threshold=15.0):
    """1回の独立したシミュレーションを実行し、最終ステップ(T=150)の態度分布のみを返す"""
    p, q = 0.03, 0.38
    _, D_matrix = generate_asymmetric_trust_network(N, m)
    opinions = np.random.uniform(-30, 30, N)
    active_agents = np.zeros(N, dtype=bool)
    c_array = np.random.uniform(0.1, 1.5, N) 
    
    for t in range(steps):
        F_t = np.sum(active_agents) / N
        for i in range(N):
            if not active_agents[i]:
                if np.random.rand() < (p + q * F_t):
                    active_agents[i] = True
                    
        A_base = 15.0 if 30 <= t <= 80 else 0.0
        opinions = update_opinions_stochastic(
            opinions, D_matrix, active_agents, A_base, 
            alpha=0.1, c_array=c_array, noise_std=noise_std, 
            bubble_factor=bubble_factor, threshold=threshold, dt=0.1
        )
    
    active_mask = active_agents
    final_opinions = opinions[active_mask]
    final_positive = np.sum(final_opinions > 1.0)
    final_negative = np.sum(final_opinions < -1.0)
    
    return final_positive, final_negative

# ==========================================
# 3. 多重試行と統計的検定の実行
# ==========================================

def run_cross_sectional_experiment(trials=30, out_dir="CrossSectional_TTest_Results"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    print(f"各条件で独立して {trials} 回のシミュレーションを実行します...")
    
    # 比較する2つの条件
    condition_A = 0.0  # フィルターバブル完全遮断
    condition_B = 1.0  # フィルターバブルなし
    
    results_A = []
    results_B = []
    
    # 条件Aの多重試行
    for i in range(trials):
        pos, neg = run_single_trial_final_state(bubble_factor=condition_A)
        results_A.append({'Trial': i+1, 'Condition': f'b={condition_A}', 'Positive': pos, 'Negative': neg})
        
    # 条件Bの多重試行
    for i in range(trials):
        pos, neg = run_single_trial_final_state(bubble_factor=condition_B)
        results_B.append({'Trial': i+1, 'Condition': f'b={condition_B}', 'Positive': pos, 'Negative': neg})
        
    # データを結合してDataFrameに変換
    df_results = pd.DataFrame(results_A + results_B)
    df_results.to_csv(os.path.join(out_dir, 'cross_sectional_data.csv'), index=False)
    
    # --- 統計的検定 (Welch's t-test) の実行 ---
    pos_A = df_results[df_results['Condition'] == f'b={condition_A}']['Positive']
    pos_B = df_results[df_results['Condition'] == f'b={condition_B}']['Positive']
    
    # equal_var=False を指定することで Welchのt検定 になる
    t_stat, p_value = stats.ttest_ind(pos_A, pos_B, equal_var=False)
    
    mean_A = pos_A.mean()
    mean_B = pos_B.mean()
    std_A = pos_A.std()
    std_B = pos_B.std()
    
    print("\n=== 統計的検定結果 (Welch's t-test) ===")
    print(f"Condition A (b={condition_A}): Mean={mean_A:.2f}, Std={std_A:.2f}")
    print(f"Condition B (b={condition_B}): Mean={mean_B:.2f}, Std={std_B:.2f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    # 結果をテキストファイルに保存
    with open(os.path.join(out_dir, 't_test_summary.txt'), 'w') as f:
        f.write(f"Hypothesis Testing: Independent Samples t-test (Welch's)\n")
        f.write(f"Trials per condition: {trials}\n\n")
        f.write(f"Condition A (b={condition_A}): Mean={mean_A:.2f}, Std={std_A:.2f}\n")
        f.write(f"Condition B (b={condition_B}): Mean={mean_B:.2f}, Std={std_B:.2f}\n")
        f.write(f"t-statistic: {t_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4e}\n")
        if p_value < 0.05:
            f.write("Conclusion: Significant difference exists between the two conditions (p < 0.05).\n")
        else:
            f.write("Conclusion: No significant difference found.\n")

    # --- 箱ひげ図 (Boxplot) の作成 ---
    plt.figure(figsize=(8, 6))
    df_results.boxplot(column='Positive', by='Condition', grid=False)
    plt.title('Final Positive Agents Distribution (T=150)')
    plt.suptitle('') # pandasのboxplotが自動生成するタイトルを消去
    plt.xlabel('Filter Bubble Condition')
    plt.ylabel('Number of Positive Agents')
    
    # グラフ内にp値を記載
    significance_text = f"Welch's t-test\n$p = {p_value:.2e}$"
    plt.text(1.5, pos_B.max(), significance_text, ha='center', va='bottom', 
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    
    plt.savefig(os.path.join(out_dir, 'boxplot_ttest.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[{out_dir}] にデータ、検定結果のサマリー、箱ひげ図を保存しました。")

def zip_and_download(folder_path, zip_name):
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(folder_path):
            for file in files_in_dir:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                
    print(f"{zip_name} を作成しました。")
    if IN_COLAB:
        files.download(zip_name)
        print("ダウンロードを開始します...")

# ==========================================
# 4. メイン実行
# ==========================================
if __name__ == "__main__":
    OUT_DIR = "TTest_Results"
    # 十分な検出力を得るため、最低30回（推奨は50〜100回）実行します。
    # ※試行回数が多いと実行に数分かかります。
    run_cross_sectional_experiment(trials=30, out_dir=OUT_DIR)
    zip_and_download(folder_path=OUT_DIR, zip_name="TTest_Results.zip")
