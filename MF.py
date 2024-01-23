import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

class MatrixFactorization:
    def __init__(self, ratings_data, n_factors=50, regularization=0.01, iterations=50):
        # 获取唯一的用户ID和物品ID
        self.user_ids = ratings_data['user_id'].unique()
        self.item_ids = ratings_data['item_id'].unique()

        # 构建用户和物品ID到索引的映射
        self.user_index_dict = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_index_dict = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        # 构建用户-物品评分矩阵
        self.ratings_matrix = self._build_ratings_matrix(ratings_data)

        # 使用ALS进行矩阵分解
        self.model = AlternatingLeastSquares(factors=n_factors, regularization=regularization, iterations=iterations)
        self.model.fit(self.ratings_matrix)

    def _build_ratings_matrix(self, ratings_data):
        # 构建稀疏矩阵
        row = [self.user_index_dict[user_id] for user_id in ratings_data['user_id']]
        col = [self.item_index_dict[item_id] for item_id in ratings_data['item_id']]
        data = ratings_data['score'].tolist()

        return coo_matrix((data, (row, col)), shape=(len(self.user_ids), len(self.item_ids)))

    def recommend_items(self, user_id, num_recommendations=10):
        user_index = self.user_index_dict.get(user_id, None)
        if user_index is None:
            return []

        user_factors = self.model.user_factors[user_index]
        
        # 使用getrow()方法获取用户的评分行
        user_row = self.ratings_matrix.getrow(user_index)

        # 找到未评分的物品（评分为0的物品）
        unrated_items = np.where(user_row.toarray().flatten() == 0)[0]

        # 计算所有物品的分数
        scores = np.dot(self.model.item_factors, user_factors)

        # 获取未评分物品的推荐结果
        recommendations = [(item_id, scores[item_index]) for item_id, item_index in self.item_index_dict.items() if item_index in unrated_items]

        # 按分数降序排序推荐结果
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:num_recommendations]

# 生成示例数据的函数
def generate_ratings_data(n_users=100, n_items=200, n_ratings=20000, random_seed=0):
    np.random.seed(random_seed)

    user_ids = np.random.choice(a=range(0, n_users), size=n_ratings, replace=True)
    item_ids = np.random.choice(a=range(0, n_items), size=n_ratings, replace=True)
    ratings = np.random.uniform(low=1, high=5, size=n_ratings)

    rating_matrix = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'score': ratings
    })

    return rating_matrix

if __name__ == "__main__":
    random_seed = 0
    n_users = 100
    n_items = 1000
    n_ratings = 20000
    num_recommendations = 10

    # 生成示例数据
    rating_matrix = generate_ratings_data(
        n_users=n_users, 
        n_items=n_items, 
        n_ratings=n_ratings, 
        random_seed=random_seed)

    # 初始化矩阵分解模型
    mf_model = MatrixFactorization(rating_matrix)

    # 为用户推荐物品
    user_id = np.random.choice(mf_model.user_ids)
    recommendations = mf_model.recommend_items(user_id, num_recommendations=num_recommendations)

    print(f"\n为用户 {user_id} 推荐的物品:")
    for item_id, score in recommendations:
        print(f"Item {item_id}: Score {score}")
