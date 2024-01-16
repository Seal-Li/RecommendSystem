# 导入所需的库
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# 定义协同过滤类
class CollaborativeFiltering:
    def __init__(self, ratings_data):
        # 初始化类的属性
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None

        # 构建用户-商品矩阵和用户-用户、商品-商品相似性矩阵
        self.user_index_dict, self.item_index_dict, self.user_item_matrix = self._build_matrix(ratings_data)
        self.item_similarity_matrix = self._calculate_item_similarity()
        self.user_similarity_matrix = self._calculate_user_similarity()

    def _build_matrix(self, ratings_data):
        # 创建用户-商品矩阵
        rating_data = ratings_data.copy()

        # 创建用户和商品的索引映射
        user_to_index = {user: i for i, user in enumerate(ratings_data['user_id'].unique())}
        item_to_index = {item: i for i, item in enumerate(ratings_data['item_id'].unique())}

        # 将用户和商品的ID替换为索引
        rating_data['user_id'].replace(user_to_index, inplace=True)
        rating_data['item_id'].replace(item_to_index, inplace=True)

        # 创建稀疏矩阵
        user_item_matrix = csr_matrix((rating_data['score'], 
                        (rating_data['user_id'], rating_data['item_id'])), dtype=np.float32)
        
        return user_to_index, item_to_index, user_item_matrix

    def user_based_recommendation(self, target_user_id, num_recommendations=10):
        # 基于用户的推荐方法
        # 获取目标用户的索引
        target_user_index = self.user_index_dict.get(target_user_id, None)
        if target_user_index is None:
            return []
        
        # 获取相似用户
        sorted_users_similar = np.argsort(self.user_similarity_matrix[target_user_index])
        most_similar_users = sorted_users_similar[-num_recommendations-1:-1][::-1]

        # 获取目标用户已经评分过的商品
        mask_target_user_seen = self.user_item_matrix[target_user_index] !=0

        # 创建推荐字典，用来保存每个商品的估计评分
        recommendations = {}
        for user_index in most_similar_users:
            mask_similar_user_seen = self.user_item_matrix[user_index] !=0
            scores = self.user_item_matrix[user_index].data
            
            # 对于与目标用户相似的每一个用户，计算他们评分过的商品的估计评分
            for item_index, score in zip(mask_similar_user_seen.indices, scores):
                # 如果目标用户已经评分过，那么就跳过
                if mask_target_user_seen[0, item_index] != 0:
                    continue
                # 保存估计评分，使用了相似性加权
                if item_index in recommendations:
                    recommendations[item_index] += score * self.user_similarity_matrix[target_user_index, user_index]
                else:
                    recommendations[item_index] = score * self.user_similarity_matrix[target_user_index, user_index]
        
        # 排序并返回前N个推荐
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:num_recommendations]

    def item_based_recommendation(self, target_user_id, target_item_id, num_recommendations=10):
        # 基于商品的推荐方法
        target_item_index = self.item_index_dict.get(target_item_id, None)
        target_user_index = self.user_index_dict.get(target_user_id, None)

        if target_item_index is None or target_user_index is None:
            return []

        # 获取与目标商品最相似的商品
        sorted_items_similar = np.argsort(self.item_similarity_matrix[target_item_index])
        most_similar_items = sorted_items_similar[-num_recommendations-1:-1][::-1]
        
        recommendations = {}

        for item_index in most_similar_items:
            # 如果目标用户已经对这个商品评分，那么就跳过
            if self.user_item_matrix[target_user_index, item_index] != 0:
                continue
            # 保存估计评分
            recommendations[item_index] = self.item_similarity_matrix[target_item_index, item_index]

        # 排序并返回前N个推荐
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return sorted_recommendations[:num_recommendations]

    def _calculate_user_similarity(self):
        # 计算并返回用户-用户相似性矩阵
        similarity_matrix = cosine_similarity(self.user_item_matrix)
        np.fill_diagonal(similarity_matrix, 0)  
        return similarity_matrix

    def _calculate_item_similarity(self):
        # 计算并返回商品-商品相似性矩阵
        item_ratings = self.user_item_matrix.T
        similarity_matrix = cosine_similarity(item_ratings)
        np.fill_diagonal(similarity_matrix, 0)  
        return similarity_matrix

def generate_ratings_data(n_users=100, n_items=200, n_ratings=20000, random_seed=0):
    # 随机生成评分数据
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

def print_recommendations(recommendations, title):
    # 打印推荐结果
    print("\n" + title)
    if len(recommendations) == 0:
        print("No recommendations found.")
    else:
        for item_id, rating in recommendations:
            print(f"Item {item_id}: Score {round(rating, 2)}")

if __name__ == "__main__":
    # 生成评分矩阵
    rating_matrix = generate_ratings_data()

    # 创建类的实例
    cf = CollaborativeFiltering(rating_matrix)

    # 生成并打印用户推荐
    user_recommendations = cf.user_based_recommendation(np.random.choice(list(cf.user_index_dict.keys())), num_recommendations=5)
    print_recommendations(user_recommendations, "基于用户的协同过滤推荐:")

    target_user_id = np.random.choice(list(cf.user_index_dict.keys()))
    target_item_id = np.random.choice(list(cf.item_index_dict.keys()))

    # 生成并打印商品推荐
    item_recommendations = cf.item_based_recommendation(target_user_id, target_item_id, num_recommendations=5)
    print_recommendations(item_recommendations, "基于物品的协同过滤推荐:")
