import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# 定义矩阵分解类
class MatrixFactorization:
    def __init__(self, ratings_data, n_factors=10, regularization=0.01, learning_rate=0.001, iterations=10):
        # 获取唯一的用户ID和物品ID
        self.user_ids = ratings_data['user_id'].unique()
        self.item_ids = ratings_data['item_id'].unique()

        # 构造用户ID与用户索引的映射字典，物品ID与物品索引的映射字典
        self.user_index_dict = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_index_dict = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

        # 根据评分数据构建评分矩阵
        self.ratings_matrix = self._build_ratings_matrix(ratings_data)

        # 使用随机正态分布初始化用户因子和物品因子
        self.user_factors = np.random.normal(size=(len(self.user_ids), n_factors))
        self.item_factors = np.random.normal(size=(len(self.item_ids), n_factors))

        # 设置超参数
        self.n_factors = n_factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.iterations = iterations

        # 训练模型
        self.train()

    def _build_ratings_matrix(self, ratings_data):
        # 根据评分数据构建稀疏的评分矩阵
        row = [self.user_index_dict[user_id] for user_id in ratings_data['user_id']]
        col = [self.item_index_dict[item_id] for item_id in ratings_data['item_id']]
        data = ratings_data['score'].tolist()

        return np.array(coo_matrix((data, (row, col)), shape=(len(self.user_ids), len(self.item_ids))).toarray())

    def _sigmoid(self, x):
        # 定义sigmoid函数，用来把数值限制在0到1之间
        return 1 / (1 + np.exp(-x))

    def train(self):
        # 训练模型，使用随机梯度下降
        for _ in range(self.iterations):
            scores_predicted = np.dot(self.user_factors, self.item_factors.T)
            diff = self.ratings_matrix - scores_predicted
            for i in range(len(self.user_ids)):
                for j in range(len(self.item_ids)):
                    # 只对评分值不为0的数据进行更新
                    if self.ratings_matrix[i, j] > 0:
                        self.user_factors[i] -= -self.learning_rate * (diff[i, j] * self.item_factors[j] + self.regularization * self.user_factors[i])
                        self.item_factors[j] -= -self.learning_rate * (diff[i, j] * self.user_factors[i] + self.regularization * self.item_factors[j])

    def recommend_items(self, user_id, num_recommendations=10):
        # 对于给定的用户，找出推荐评分最高的前N个物品
        user_index = self.user_index_dict.get(user_id, None)
        if user_index is None:
            return []

        user_factors = self.user_factors[user_index, :]
        scores = self._sigmoid(np.dot(user_factors, self.item_factors.T)) * 5   # 用sigmoid函数把评分调整到0-5之间

        unrated_items = np.where(self.ratings_matrix[user_index, :] == 0)[0]

        recommendations = [
            (item_id, scores[item_index]) 
            for item_id, item_index in self.item_index_dict.items() 
            if item_index in unrated_items]
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:num_recommendations]


def generate_ratings_data(n_users=100, n_items=200, n_ratings=20000, random_seed=0):
    # 生成评分数据
    np.random.seed(random_seed)
    user_ids = np.random.choice(a=range(0, n_users), size=n_ratings, replace=True)
    item_ids = np.random.choice(a=range(0, n_items), size=n_ratings, replace=True)
    ratings = np.random.uniform(low=1, high=5, size=n_ratings)
    rating_matrix = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'score': ratings})

    return rating_matrix


if __name__ == "__main__":
    # 主函数，生成数据并调用矩阵分解模型
    random_seed = 0
    n_users = 100
    n_items = 10000
    n_ratings = 20000

    n_factors=10
    regularization=0.01
    learning_rate=0.001
    iterations=10

    num_recommendations = 10
    rating_matrix = generate_ratings_data(
        n_users=n_users, 
        n_items=n_items, 
        n_ratings=n_ratings, 
        random_seed=random_seed)

    mf_model = MatrixFactorization(
        rating_matrix, 
        n_factors=n_factors, 
        regularization=regularization, 
        learning_rate=learning_rate, 
        iterations=iterations)

    user_id = np.random.choice(mf_model.user_ids)
    recommendations = mf_model.recommend_items(
        user_id, 
        num_recommendations=num_recommendations)

    print(f"为用户 {user_id} 推荐的物品:")
    for item_id, score in recommendations:
        print(f"物品 {item_id}: 评分 {score}")
