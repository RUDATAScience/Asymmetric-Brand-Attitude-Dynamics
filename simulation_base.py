import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
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
# 1. モデル定義・シミュレーション関数
# ==========================================

def generate_asymmetric_trust_network(N=300, m=3):
    """スケールフリーネットワークを用いた非対称な信頼行列の生成"""
    G = nx.barabasi_albert_graph(N, m)
    adj_matrix = nx.to_numpy_array(G)
    D_matrix = np.random.uniform(-1, 1, (N, N))
    D_matrix = D_matrix * adj_matrix
    return G, D_matrix

def cutoff_function(I_i, I_j, threshold=15.0):
    """意見の差による影響の減衰（シグモイド関数）"""
    diff = np.abs(I_i - I_j)
    return 1.0 / (1.0 + np.exp(diff - threshold))

def update_opinions(I, D, active_agents, A_t, alpha, c_array, dt=0.1):
    """TDMに基づく意見の更新"""
    N = len(I)
    delta_I = np.zeros(N)
    active_indices = np.where(active_agents)[0]
    
    for i in active_indices:
        term1 = -alpha * I[i]
        term2 = c_array[i] * A_t
        term3 = 0
        for j in active_indices:
            if D[i, j] != 0:
                phi = cutoff_function(I[i], I[j])
                term3 += D[i, j] * phi * (I[j] - I[i])
        delta_I[i] = (term1 + term2 + term3) * dt
        
    return I + delta_I

def run_simulation(steps=100, p=0.03, q=0.38):
    """シミュレーションの実行（Bassモデルの参加プロセス統合）"""
    N = 300
    G, D_matrix = generate_asymmetric_trust_network(N)
    opinions = np.random.uniform(-30, 30, N)
    active_agents = np.zeros(N, dtype=bool)
    c_array = np.random.uniform(-0.5, 1.5, N) 
    
    history_opinions = []
    history_active = []
    
    for t in range(steps):
        F_t = np.sum(active_agents) / N
        
        for i in range(N):
            if not active_agents[i]:
                P_entry = p + q * F_t
                if np.random.rand() < P_entry:
                    active_agents[i] = True
                    
        # 外部影響（例: t=20から40まで広告投入）
        A_t = 10.0 if 20 <= t <= 40 else 0.0
        
        opinions = update_opinions(opinions, D_matrix, active_agents, A_t, alpha=0.1, c_array=c_array)
        
        # 記録（参加していないエージェントの意見は便宜上NaNや0ではなく現状の値を保持）
        history_opinions.append(opinions.copy())
        history_active.append(active_agents.copy())
        
    return np.array(history_opinions), np.array(history_active)

# ==========================================
# 2. 結果の可視化とファイル保存
# ==========================================

def export_results(history_opinions, history_active, output_dir="simulation_results"):
    """結果のプロット生成とCSV保存"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    steps, N = history_opinions.shape
    
    # グラフ1: 個別エージェントの意見推移 (Opinion Dynamics)
    plt.figure(figsize=(10, 6))
    for i in range(N):
        # 参加後のデータのみプロットするなどの工夫も可能ですが、ここでは全体をプロット
        plt.plot(range(steps), history_opinions[:, i], color='green', alpha=0.1)
    plt.title('Opinion Dynamics Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Opinion Value')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'opinion_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # グラフ2: ポジティブ/ネガティブの構成比推移 (Attitude Composition)
    # アクティブなエージェントのみを集計
    positive_counts = []
    negative_counts = []
    neutral_counts = []
    
    for t in range(steps):
        active_mask = history_active[t]
        current_opinions = history_opinions[t][active_mask]
        
        positive_counts.append(np.sum(current_opinions > 0))
        negative_counts.append(np.sum(current_opinions < 0))
        neutral_counts.append(np.sum(current_opinions == 0))

    plt.figure(figsize=(10, 6))
    plt.plot(range(steps), positive_counts, label='Positive (>0)', color='red')
    plt.plot(range(steps), negative_counts, label='Negative (<0)', color='blue')
    plt.plot(range(steps), neutral_counts, label='Neutral (=0)', color='gray')
    
    # 総参加者数のライン
    total_active = np.sum(history_active, axis=1)
    plt.plot(range(steps), total_active, label='Total Active Agents', color='black', linestyle='--')
    
    plt.title('Attitude Composition Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Agents')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'attitude_composition.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # CSV出力
    df_opinions = pd.DataFrame(history_opinions, columns=[f'Agent_{i}' for i in range(N)])
    df_opinions.index.name = 'TimeStep'
    df_opinions.to_csv(os.path.join(output_dir, 'opinion_history.csv'))
    
    df_composition = pd.DataFrame({
        'Positive': positive_counts,
        'Negative': negative_counts,
        'Neutral': neutral_counts,
        'Total_Active': total_active
    })
    df_composition.index.name = 'TimeStep'
    df_composition.to_csv(os.path.join(output_dir, 'attitude_composition.csv'))
    
    print(f"[{output_dir}] ディレクトリにCSVとPNGを保存しました。")

def zip_and_download(folder_path="simulation_results", zip_name="simulation_results.zip"):
    """フォルダをZIP化してダウンロード"""
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(folder_path):
            for file in files_in_dir:
                file_path = os.path.join(root（２）ile)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                
    print(f"{zip_name} を作成しました。")
    
    if IN_COLAB:
        files.download(zip_name)
        print("ダウンロードを開始します...")

# ==========================================
# 3. 実行メイン処理
# ==========================================
if __name__ == "__main__":
    print("シミュレーションを実行中...")
    # パラメータは論文のCase Aに準拠 (p=0.03, q=0.38)
    history_op, history_act = run_simulation(steps=150, p=0.03, q=0.38)
    
    print("結果を可視化・エクスポート中...")
    out_dir = "TDM_Bass_Experiment"
    export_results(history_op, history_act, output_dir=out_dir)
    
    zip_and_download(folder_path=out_dir, zip_name="TDM_Bass_Experiment.zip")
