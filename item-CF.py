from pyspark.sql.functions import col, collect_set, size, expr
from pyspark.sql.functions import broadcast
from pyspark.sql.types import FloatType
from pyspark.sql.functions import array
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.feature import HashingTF
from pyspark import StorageLevel
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_spark_session():
    spark = (
        SparkSession.builder.appName("FullRecruit ItemCF")
        .enableHiveSupport()
        .getOrCreate()
    )
    return spark


def load_data(start_date, end_date, format_date):
    action_sql = f"""
        select 
            uuid, item_id, rating
        from log_table
    """
    item_sql = f"""
        select 
            item_id, salary, 
            cate1, cate2, cate3,
            city_id as local1, 
            district_id as local2,
            block_id as local3,
            cast(split(salary, '-')[0] as int) as salary_lower, 
            cast(split(salary, '-')[1] as int) as salary_upper
        from item_table
    """

    logging.info(f"user-item action sql:\n{action_sql}")
    interaction = spark.sql(action_sql).repartition(2000)

    logging.info(f"item sql:\n{item_sql}")
    item = spark.sql(item_sql).repartition(200)
    return interaction, item


def get_jaccard_metrics(interaction):
    user_count = interaction.select("uuid").distinct().count()
    item_user_df = interaction.groupBy("item_id").agg(
        collect_set("uuid").alias("user_list")
    )
    hashingTF = HashingTF(
        inputCol="user_list", outputCol="features", numFeatures=user_count
    )
    item_user_vector = hashingTF.transform(item_user_df)
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(item_user_vector)
    similarity = (
        model.approxSimilarityJoin(
            item_user_vector,
            item_user_vector,
            threshold=0.9,
            distCol="jaccard_distance",
        )
        .filter(col("datasetA.item_id") != col("datasetB.item_id"))
        .select(
            col("datasetA.item_id").alias("item_id"),
            col("datasetB.item_id").alias("similar_item_id"),
            col("datasetA.features").alias("item_features"),
            col("datasetB.features").alias("similar_item_features"),
            "jaccard_distance",
        )
    )
    return similarity


def similar_filter(similarity, item):
    item1 = item.alias("item1")
    item2 = item.alias("item2")
    similarity_info = (
        similarity.alias("similarity")
        .join(item1, similarity.item_id == item1.item_id)
        .select(
            col("similarity.item_id").alias("item_id"),
            col("similarity.similar_item_id").alias("similar_item_id"),
            col("similarity.item_features").alias("item_features"),
            col("similarity.similar_item_features").alias("similar_item_features"),
            col("similarity.jaccard_distance").alias("jaccard_distance"),
            col("item1.cate3").alias("cate3"),
            col("item1.local1").alias("local1"),
            col("item1.local2").alias("local2"),
            col("item1.salary_lower").alias("salary_lower"),
            col("item1.salary_upper").alias("salary_upper"),
        )
    )
    similarity_info = (
        similarity_info.alias("similarity_info")
        .join(item2, similarity_info.similar_item_id == item2.item_id)
        .select(
            col("similarity_info.item_id").alias("item_id"),
            col("similarity_info.similar_item_id").alias("similar_item_id"),
            col("similarity_info.item_features").alias("item_features"),
            col("similarity_info.similar_item_features").alias("similar_item_features"),
            col("similarity_info.jaccard_distance").alias("jaccard_distance"),
            col("similarity_info.cate3").alias("cate3"),
            col("similarity_info.local1").alias("local1"),
            col("similarity_info.local2").alias("local2"),
            col("similarity_info.salary_lower").alias("salary_lower"),
            col("similarity_info.salary_upper").alias("salary_upper"),
            col("item2.cate3").alias("similar_cate3"),
            col("item2.local1").alias("similar_local1"),
            col("item2.local2").alias("similar_local2"),
            col("item2.salary_lower").alias("similar_salary_lower"),
            col("item2.salary_upper").alias("similar_salary_upper"),
        )
    )

    similarity_info = similarity_info.filter(
        (col("cate3") == col("similar_cate3"))
        & (col("local1") == col("similar_local1"))
        & (col("local2") == col("similar_local2"))
        & (col("similar_salary_lower") >= col("salary_lower") * 0.75)
        & (col("similar_salary_upper") <= col("salary_upper") * 1.25)
    ).drop(
        "cate3",
        "local1",
        "local2",
        "salary_lower",
        "salary_upper",
        "similar_cate3",
        "similar_local1",
        "similar_local2",
        "similar_salary_lower",
        "similar_salary_upper",
    )

    window_spec = Window.partitionBy("item_id").orderBy("jaccard_distance")
    similar_items = (
        similarity_info.withColumn("rank", F.row_number().over(window_spec))
        .filter(F.col("rank") <= 50)
        .drop("rank", "jaccard_distance")
    )

    return similar_items


def get_recommendations(interaction, similar_items):
    def cosine_similarity(vecA, vecB):
        dot_product = float(vecA.dot(vecB))
        normA = float(vecA.norm(2))
        normB = float(vecB.norm(2))
        if normA > 0 and normB > 0:
            return dot_product / (normA * normB)
        else:
            return 0.0

    cosine_similarity_udf = F.udf(cosine_similarity, FloatType())
    recommendations = (
        interaction.join(similar_items, on="item_id", how="inner")
        .select(
            "uuid",
            "rating",
            "item_id",
            "item_features",
            "similar_item_id",
            "similar_item_features",
        )
        .withColumn(
            "cosine_similarity",
            cosine_similarity_udf(col("item_features"), col("similar_item_features")),
        )
        .withColumn("score", col("rating") * col("cosine_similarity"))
        .select("uuid", "similar_item_id", "score")
        .groupBy("uuid", "similar_item_id")
        .agg(F.sum("score").alias("total_score"))
    )

    recommendations = recommendations.select(
        "uuid",
        col("similar_item_id").alias("post_id"),
        col("total_score").alias("score"),
    )

    return recommendations


if __name__ == "__main__":
    date = sys.argv[1]
    date_obj = datetime.strptime(date, "%Y%m%d")
    format_date = date_obj.strftime("%Y-%m-%d")
    start_date = (date_obj - timedelta(days=31)).strftime("%Y%m%d")

    logging.info(f"start_date: {start_date}, date: {date}, format_date: {format_date}")
    spark = get_spark_session()
    interaction, item = load_data(start_date, date, format_date)
    item.cache()
    logging.info(f"item count: {item.count()}")

    similarity = get_jaccard_metrics(interaction)
    similar_items = similar_filter(similarity, item)
    recommendations = get_recommendations(interaction, similar_items)
    recommendations.createOrReplaceTempView("recommendations")

    insert_sql = f"""
        insert overwrite table table_name partition(dt='{date}')
        select uuid, post_id, score from recommendations
    """
    logging.info(f"insert_sql:\n{insert_sql}")
    spark.sql(insert_sql)
    spark.stop()
