# User Matching System using QUBO and D-Wave
# ユーザーのトピック関心度を基にした最適マッチングシステム

import sys
import os
import numpy as np
import pandas as pd
import json
import itertools
from typing import List, Dict, Tuple, Optional
from dimod import BinaryQuadraticModel, ExactSolver
from dwave.samplers import SimulatedAnnealingSampler

# Django設定の初期化
if not hasattr(sys, '_django_setup_done'):
    django_project_root = os.path.join(os.path.dirname(__file__), '..', '..', 'webapp')
    sys.path.insert(0, django_project_root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'project.settings')
    
    import django
    django.setup()
    sys._django_setup_done = True

# Django関連のインポート（setup後に実行）
try:
    from accounts.models import User, UserTopicScore
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    print("警告: Django環境が利用できません。モックデータのみで動作します。")

class UserMatcher:
    def __init__(self):
        self.japanese_labels = ["社会", "まなび", "テクノロジー", "カルチャー", "アウトドア", "フード", 
                               "旅行おでかけ", "ライフスタイル", "ビジネス", "読書", "キャリア", 
                               "デザイン", "IT", "経済投資", "ネットワーク"]
        
        self.english_labels = ["society", "learning", "technology", "culture", "outdoor", "food", 
                              "travel", "lifestyle", "business", "reading", "career", 
                              "design", "IT", "economics", "network"]
        
        # ラベルマッピング辞書（英語→日本語、日本語→英語）
        self.eng_to_jp = dict(zip(self.english_labels, self.japanese_labels))
        self.jp_to_eng = dict(zip(self.japanese_labels, self.english_labels))
        
        # デフォルトは日本語ラベルを使用
        self.topic_labels = self.japanese_labels
        
    def load_user_data(self, file_path: str) -> pd.DataFrame:
        """ユーザーデータを読み込み"""
        return pd.read_csv(file_path)
    
    def load_user_data_from_database(self, min_score_threshold: float = 0.01) -> pd.DataFrame:
        """データベースからユーザーデータを読み込み"""
        if not DJANGO_AVAILABLE:
            raise ImportError("Django環境が利用できません。データベースからのデータ読み込みはできません。")
        
        # スコアがmin_score_threshold以上のトピックを持つユーザーを取得
        users_with_scores = User.objects.filter(
            topic_scores__score__gte=min_score_threshold
        ).distinct()
        
        users_data = []
        
        for user in users_with_scores:
            # ユーザーのトピックマトリックスを取得
            topic_matrix = UserTopicScore.get_user_topic_matrix(user)
            
            # データフレーム用の行を作成
            user_data = {'user_id': user.account_id}
            user_data.update(topic_matrix)
            users_data.append(user_data)
        
        df = pd.DataFrame(users_data)
        print(f"データベースから {len(users_data)} 人のユーザーデータを読み込みました")
        
        return df
    
    def filter_active_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析対象ユーザーをフィルタリング（何かしらのトピックに0以上の値があるユーザー）"""
        topic_cols = [col for col in df.columns if col in self.topic_labels]
        # 各ユーザーについて、トピックスコアの合計が0より大きいかチェック
        df['total_score'] = df[topic_cols].sum(axis=1)
        active_users = df[df['total_score'] > 0].copy()
        return active_users.drop('total_score', axis=1)
    
    def calculate_compatibility_score(self, user1_scores: pd.Series, user2_scores: pd.Series) -> float:
        """2人のユーザー間の相性スコアを計算"""
        topic_cols = [col for col in user1_scores.index if col in self.topic_labels]
        
        # ベクトルの内積を使用（共通の関心が高いほど高スコア）
        dot_product = sum(user1_scores[col] * user2_scores[col] for col in topic_cols)
        
        # 正規化のためにベクトルの大きさを計算
        norm1 = np.sqrt(sum(user1_scores[col]**2 for col in topic_cols))
        norm2 = np.sqrt(sum(user2_scores[col]**2 for col in topic_cols))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # コサイン類似度を返す
        return dot_product / (norm1 * norm2)
    
    def create_qubo_matrix(self, users_df: pd.DataFrame) -> Tuple[Dict, List[str]]:
        """QUBOマトリックスを作成"""
        n_users = len(users_df)
        user_ids = users_df['user_id'].tolist()
        
        # 変数定義: x_{i,j} = 1 if user i is matched with user j, 0 otherwise
        # 対称性を保つため、i < j の組み合わせのみ考慮
        Q = {}
        
        # すべてのユーザーペアに対して相性スコアを計算
        for i in range(n_users):
            for j in range(i+1, n_users):  # i < j
                user1 = users_df.iloc[i]
                user2 = users_df.iloc[j]
                compatibility = self.calculate_compatibility_score(user1, user2)
                
                # 負の値にして最小化問題に変換（相性が高いペアを選びたいため）
                var_name = f"x_{i}_{j}"
                Q[(var_name, var_name)] = -compatibility
        
        # 制約: 各ユーザーは最大1回しかマッチングされない
        # ペナルティ項を追加
        penalty_weight = 10.0  # 制約違反に対する重いペナルティ
        
        for i in range(n_users):
            # ユーザーiが関わるすべてのペア変数
            involving_i = []
            for j in range(n_users):
                if i < j:
                    involving_i.append(f"x_{i}_{j}")
                elif j < i:
                    involving_i.append(f"x_{j}_{i}")
            
            # ペナルティ項: (sum of variables involving user i - 1)^2
            # 展開すると: sum_k x_k^2 + 2*sum_{k<l} x_k*x_l - 2*sum_k x_k + 1
            for var in involving_i:
                Q[(var, var)] = Q.get((var, var), 0) + penalty_weight
                
            for k, var1 in enumerate(involving_i):
                for var2 in involving_i[k+1:]:
                    key = tuple(sorted([var1, var2]))
                    Q[key] = Q.get(key, 0) + 2 * penalty_weight
                    
            for var in involving_i:
                Q[(var, var)] = Q.get((var, var), 0) - 2 * penalty_weight
        
        return Q, user_ids
    
    def solve_matching(self, Q: Dict, use_quantum: bool = False) -> Dict:
        """QUBOを解いてマッチング結果を取得"""
        bqm = BinaryQuadraticModel.from_qubo(Q)
        
        if use_quantum:
            # 実際のD-Wave量子コンピュータを使用する場合
            # from dwave.system import DWaveSampler, EmbeddingComposite
            # sampler = EmbeddingComposite(DWaveSampler())
            # sampleset = sampler.sample(bqm, num_reads=100)
            print("量子コンピュータは利用できません。シミュレーションを使用します。")
            use_quantum = False
        
        if not use_quantum:
            # シミュレーションを使用
            if len(bqm.variables) <= 20:  # 小規模問題は厳密解
                sampler = ExactSolver()
                sampleset = sampler.sample(bqm)
            else:  # 大規模問題はシミュレーテッドアニーリング
                sampler = SimulatedAnnealingSampler()
                sampleset = sampler.sample(bqm, num_reads=1000)
        
        best_solution = sampleset.first
        return best_solution.sample, best_solution.energy
    
    def interpret_solution(self, solution: Dict, user_ids: List[str]) -> List[Tuple[str, str]]:
        """解を解釈してマッチングペアを抽出"""
        matches = []
        n_users = len(user_ids)
        
        for i in range(n_users):
            for j in range(i+1, n_users):
                var_name = f"x_{i}_{j}"
                if solution.get(var_name, 0) == 1:
                    matches.append((user_ids[i], user_ids[j]))
        
        return matches
    
    def run_matching(self, users_df: pd.DataFrame, use_quantum: bool = False) -> Dict:
        """マッチングの全体的な実行"""
        print(f"=== ユーザーマッチング開始 ===")
        print(f"対象ユーザー数: {len(users_df)}")
        
        # QUBOマトリックス作成
        Q, user_ids = self.create_qubo_matrix(users_df)
        print(f"QUBO変数数: {len(Q)}")
        
        # 最適化実行
        solution, energy = self.solve_matching(Q, use_quantum)
        print(f"最適解エネルギー: {energy}")
        
        # 結果解釈
        matches = self.interpret_solution(solution, user_ids)
        print(f"マッチング数: {len(matches)}")
        
        # 結果の詳細表示
        total_compatibility = 0
        for i, (user1_id, user2_id) in enumerate(matches):
            user1_data = users_df[users_df['user_id'] == user1_id].iloc[0]
            user2_data = users_df[users_df['user_id'] == user2_id].iloc[0]
            compatibility = self.calculate_compatibility_score(user1_data, user2_data)
            total_compatibility += compatibility
            print(f"マッチング {i+1}: {user1_id} <-> {user2_id} (相性スコア: {compatibility:.4f})")
        
        print(f"総合相性スコア: {total_compatibility:.4f}")
        print(f"平均相性スコア: {total_compatibility/len(matches):.4f}" if matches else "マッチングなし")
        
        return {
            'matches': matches,
            'total_compatibility': total_compatibility,
            'average_compatibility': total_compatibility/len(matches) if matches else 0,
            'energy': energy,
            'solution': solution
        }
    
    def save_matching_results_to_database(self, matches: List[Tuple[str, str]], total_compatibility: float):
        """マッチング結果をデータベースに保存（将来の拡張用）"""
        if not DJANGO_AVAILABLE:
            print("警告: Django環境が利用できないため、結果をデータベースに保存できません。")
            return False
        
        # 将来的にマッチング結果を保存するテーブルを作成する場合の準備
        print(f"マッチング結果: {len(matches)}組、総合相性: {total_compatibility:.4f}")
        print("注意: マッチング結果の永続化機能は未実装です。")
        return True
    
    def run_database_matching(self, min_score_threshold: float = 0.01, use_quantum: bool = False, save_results: bool = False) -> Dict:
        """データベースからユーザーデータを読み込んでマッチングを実行"""
        print(f"=== データベースベースマッチング開始 ===")
        
        try:
            # データベースからユーザーデータを読み込み
            users_df = self.load_user_data_from_database(min_score_threshold)
            
            if len(users_df) == 0:
                print("マッチング対象のユーザーが見つかりませんでした。")
                return {'matches': [], 'total_compatibility': 0, 'average_compatibility': 0}
            
            # アクティブユーザーのフィルタリング
            active_users = self.filter_active_users(users_df)
            
            if len(active_users) < 2:
                print(f"マッチング対象のアクティブユーザーが不足しています（現在: {len(active_users)}人、必要: 2人以上）")
                return {'matches': [], 'total_compatibility': 0, 'average_compatibility': 0}
            
            # マッチング実行
            result = self.run_matching(active_users, use_quantum)
            
            # 結果の保存（オプション）
            if save_results and result['matches']:
                self.save_matching_results_to_database(result['matches'], result['total_compatibility'])
            
            return result
            
        except Exception as e:
            print(f"データベースマッチング中にエラーが発生しました: {e}")
            print("モックデータでのテストに切り替えます...")
            return self.run_mock_matching(use_quantum)

def create_mock_data(n_users: int = 10, output_file: str = "mock_user_data.csv") -> pd.DataFrame:
    """モックデータを作成"""
    np.random.seed(42)  # 再現性のため
    
    # 改善されたラベル（高スコア獲得のため6個に削減）
    topic_labels = ["社会", "まなび", "テクノロジー", "カルチャー", "アウトドア", "フード", 
                   "旅行おでかけ", "ライフスタイル", "ビジネス", "読書", "キャリア", 
                   "デザイン", "IT", "経済投資", "ネットワーク"]
    
    users_data = []
    
    for i in range(n_users):
        user_data = {'user_id': f'user_{i+1:03d}'}
        
        # ランダムにトピックスコアを生成（一部のユーザーは特定の分野に偏向）
        if i % 3 == 0:  # テック系ユーザー
            base_scores = np.random.exponential(0.3, len(topic_labels))
            base_scores[topic_labels.index("テクノロジー")] *= 3
            base_scores[topic_labels.index("まなび")] *= 2
        elif i % 3 == 1:  # カルチャー系ユーザー
            base_scores = np.random.exponential(0.3, len(topic_labels))
            base_scores[topic_labels.index("カルチャー")] *= 3
            base_scores[topic_labels.index("フード")] *= 2
        else:  # バランス型ユーザー
            base_scores = np.random.exponential(0.5, len(topic_labels))
        
        # 正規化
        total_score = np.sum(base_scores)
        if total_score > 0:
            base_scores = base_scores / total_score
        
        for j, topic in enumerate(topic_labels):
            user_data[topic] = base_scores[j]
        
        users_data.append(user_data)
    
    df = pd.DataFrame(users_data)
    df.to_csv(output_file, index=False)
    print(f"モックデータを {output_file} に保存しました")
    return df

    def run_mock_matching(self, use_quantum: bool = False) -> Dict:
        """モックデータでのマッチング実行"""
        print("=== モックデータマッチング ===")
        mock_df = create_mock_data(n_users=8, output_file="fallback_mock.csv")
        active_users = self.filter_active_users(mock_df)
        return self.run_matching(active_users, use_quantum) 