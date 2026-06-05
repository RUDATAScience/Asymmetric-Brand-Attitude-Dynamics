import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os
import zipfile
import shutil
import scipy.stats as stats

# Google Colab環境でのダウンロード用ライブラリ
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==========================================
# 1. 拡張シミュレーション基盤モデル
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
        
        # フィルターバブルの一般化：bubble_factor (0.0: 完全遮断 〜 1.0: バブルなし)
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

def run_simulation(steps=150, N=300, m=3, noise_std=3.0, bubble_factor=0.1, threshold=15.0, trials=3):
    """複数回の試行(trials)を行い、平均的な態度分布を返す"""
    p, q = 0.03, 0.38
    
    avg_positive = np.zeros(steps)
    avg_negative = np.zeros(steps)
    
    for trial in range(trials):
        _, D_matrix = generate_asymmetric_trust_network(N, m)
        opinions = np.random.uniform(-30, 30, N)
        active_agents = np.zeros(N, dtype=bool)
        c_array = np.random.uniform(0.1, 1.5, N) 
        
        pos_counts = []
        neg_counts = []
        
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
            current_opinions = opinions[active_mask]
            pos_counts.append(np.sum(current_opinions > 1.0))
            neg_counts.append(np.sum(current_opinions < -1.0))
            
        avg_positive += np.array(pos_counts)
        avg_negative += np.array(neg_counts)
        
    return avg_positive / trials, avg_negative / trials

# ==========================================
# 2. 統計解析関数（有効標本サイズとp値）
# ==========================================

def calculate_effective_statistics(time_series_x, time_series_y):
    """
    時系列データの自己相関(Lag-1)を考慮した有効標本サイズ(n_eff)と補正p値を計算する
    """
    N_nom = len(time_series_x)
    
    # ピアソンの積率相関係数
    r, _ = stats.pearsonr(time_series_x, time_series_y)
    
    # Lag-1自己相関の計算 (Pandasを使用)
    s_x = pd.Series(time_series_x)
    s_y = pd.Series(time_series_y)
    rho_x = s_x.autocorr(lag=1)
    rho_y = s_y.autocorr(lag=1)
    
    # 代表的な自己相関として平均をとる（より厳密には個別補正が必要だが近似として）
    rho_avg = (abs(rho_x) + abs(rho_y)) / 2.0
    
    # 有効標本サイズの計算 (Bartlett's formula approximation)
    if rho_avg < 1.0:
        n_eff = N_nom * ((1 - rho_avg) / (1 + rho_avg))
    else:
        n_eff = 2.0 # 相関が強すぎる場合のセーフティ
        
    # 自由度の補正とp値の再計算
    df = max(1, n_eff - 2)
    # t統計量
    if abs(r) == 1.0:
        p_val_adj = 0.0
    else:
        t_stat = r * np.sqrt(df / (1 - r**2))
        p_val_adj = stats.t.sf(np.abs(t_stat), df) * 2 # 両側検定
    
    return {
        "Nominal_N": N_nom,
        "Effective_N": round(n_eff, 2),
        "Lag1_Autocorr": round(rho_avg, 3),
        "Pearson_r": round(r, 4),
        "Adjusted_p_value": p_val_adj
    }

# ==========================================
# 3. 感度分析の実行とエクスポート
# ==========================================

def run_sensitivity_experiments(out_dir="Sensitivity_Analysis_Results"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    steps = 150
    # キャンペーン強度フラグ（相関分析用）
    campaign_flag = np.array([1.0 if 30 <= t <= 80 else 0.0 for t in range(steps)])

    print("実験1: ネットワーク結合数(m)の感度分析を開始...")
    m_list = [2, 3, 5, 10]
    plt.figure(figsize=(10, 6))
    df_m = pd.DataFrame({'Time': range(steps)})
    
    for m in m_list:
        pos, neg = run_simulation(m=m, trials=3)
        plt.plot(pos, label=f'm={m} (Positive)')
        df_m[f'm_{m}_Positive'] = pos
        df_m[f'm_{m}_Negative'] = neg
        
    plt.title("Sensitivity 1: Network Hub Influence (m parameter)")
    plt.xlabel("Time Step")
    plt.ylabel("Average Positive Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvspan(30, 80, color='yellow', alpha=0.1)
    plt.savefig(os.path.join(out_dir, 'exp1_network_m.png'))
    plt.close()
    df_m.to_csv(os.path.join(out_dir, 'exp1_network_m.csv'), index=False)

    print("実験2: フィルターバブル強度(bubble_factor)の感度分析を開始...")
    bubble_list = [0.0, 0.1, 0.5, 1.0] # 0.0:完全遮断, 1.0:バブルなし
    plt.figure(figsize=(10, 6))
    df_b = pd.DataFrame({'Time': range(steps)})
    stat_results = []
    
    for b in bubble_list:
        pos, neg = run_simulation(bubble_factor=b, trials=3)
        plt.plot(pos, label=f'BubbleFactor={b} (Positive)')
        df_b[f'b_{b}_Positive'] = pos
        df_b[f'b_{b}_Negative'] = neg
        
        # 統計量の計算（キャンペーンフラグとPositive層の増加の相関）
        stats_dict = calculate_effective_statistics(campaign_flag, pos)
        stats_dict['Bubble_Factor'] = b
        stat_results.append(stats_dict)
        
    plt.title("Sensitivity 2: Filter Bubble Factor")
    plt.xlabel("Time Step")
    plt.ylabel("Average Positive Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvspan(30, 80, color='yellow', alpha=0.1)
    plt.savefig(os.path.join(out_dir, 'exp2_filter_bubble.png'))
    plt.close()
    df_b.to_csv(os.path.join(out_dir, 'exp2_filter_bubble.csv'), index=False)
    
    # 統計結果の保存
    pd.DataFrame(stat_results).to_csv(os.path.join(out_dir, 'exp2_statistical_analysis.csv'), index=False)

    print("実験3: 意見閾値(threshold)の感度分析を開始...")
    threshold_list = [5.0, 10.0, 15.0, 30.0]
    plt.figure(figsize=(10, 6))
    df_t = pd.DataFrame({'Time': range(steps)})
    
    for th in threshold_list:
        pos, neg = run_simulation(threshold=th, trials=3)
        plt.plot(pos, label=f'Threshold={th} (Positive)')
        df_t[f'th_{th}_Positive'] = pos
        df_t[f'th_{th}_Negative'] = neg
        
    plt.title("Sensitivity 3: Opinion Tolerance Threshold")
    plt.xlabel("Time Step")
    plt.ylabel("Average Positive Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axvspan(30, 80, color='yellow', alpha=0.1)
    plt.savefig(os.path.join(out_dir, 'exp3_threshold.png'))
    plt.close()
    df_t.to_csv(os.path.join(out_dir, 'exp3_threshold.csv'), index=False)

    print("全ての実験が完了しました。ZIPファイルを作成します。")

def zip_and_download(folder_path="Sensitivity_Analysis_Results", zip_name="Sensitivity_Analysis_Results.zip"):
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
    OUT_DIR = "Sensitivity_Analysis_Results"
    # モンテカルロ試行回数を減らして高速化（本格的な論文用には trials=10~30 を推奨）
    run_sensitivity_experiments(out_dir=OUT_DIR)
    zip_and_download(folder_path=OUT_DIR, zip_name="Sensitivity_Analysis_Results.zip")
