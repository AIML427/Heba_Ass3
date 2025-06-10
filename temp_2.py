import logging
import math
import numpy as np
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, stddev, udf
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# --------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
# --------------------------------

def select_top_important_features(df, feature_cols, label_col, num_features=50):
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    
    logger.info("Starting feature selection using RandomForest feature importances...")

    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)
    
    # Fit a RandomForest for feature importance
    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=50, maxDepth=5, seed=42)
    model = rf.fit(df)
    
    # Extract importances
    importances = model.featureImportances.toArray()
    feature_importance_dict = dict(zip(feature_cols, importances))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Select top N features
    top_features = [f[0] for f in sorted_features[:num_features]]
    logger.info(f"Top {num_features} selected features: {top_features}")
    
    return top_features

def load_data(file_path, spark):
    """
    Load data from CSV (tab-delimited) into Spark DataFrame.
    """
    # Build absolute file path for Spark
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    spark_file_path = f"file:/{file_path.as_posix()}"
    
    # Read with header and infer schema
    df = spark.read.option("delimiter", "\t").option("header", "true").csv(spark_file_path, inferSchema=True)
    
    logger.info(f"Loaded data with {df.count()} rows and {len(df.columns)} columns.")
    logger.info(f"Columns: {df.columns}")
    return df

def data_prepration(df):

    # 1. DROP CATEGORICAL --------------------------
    categorical_cols = ["proto", "service"]
    df = df.drop(*categorical_cols)

    # 2. ENCODED TARGET ----------------------------
    indexer = StringIndexer(inputCol="Attack_type", outputCol="Attack_type_index", handleInvalid="keep")
    df = indexer.fit(df).transform(df)

    # 3. NUMERIC FEATURE ---------------------------
    # Identify numeric feature columns
    feature_cols = [c for c in df.columns if c not in ["Attack_type", "Attack_type_index"]]
    
    # Cast numeric columns to double
    for c in feature_cols:
        df = df.withColumn(c, col(c).cast("double"))

    # 4. MISSING VALUE -----------------------------
    #df = df.fillna(0)  # Fill missing numeric with 0

    # Drop constant columns (zero variance)
    #stats = df.select(*(stddev(col(c)).alias(c) for c in feature_cols)).collect()[0].asDict()
    #zero_var_cols = [k for k, v in stats.items() if v == 0 or v is None]
    #if zero_var_cols:
    #    logger.warning(f"Dropping zero variance columns: {zero_var_cols}")
    #    feature_cols = [c for c in feature_cols if c not in zero_var_cols]

    # ----------------------------------------------
    # Remove rows with NaN or Inf in features
        #def has_nan_inf(v):
        #    return any(math.isnan(x) or math.isinf(x) for x in v)
        #has_nan_inf_udf = udf(has_nan_inf, BooleanType())
        #invalid_count = df.filter(has_nan_inf_udf(col("features"))).count()
        #if invalid_count > 0:
        #    logger.warning(f"Dropping {invalid_count} rows with NaN/Inf in features")
        #    df = df.filter(~has_nan_inf_udf(col("features")))

    # 5. HANDLE IMBALANCE: Compute class weights
    label_counts = df.groupBy("Attack_type_index").count().collect()
    total = df.count()
    class_weights = {row["Attack_type_index"]: total / (len(label_counts) * row["count"]) for row in label_counts}

    # Add class_weight column
    from pyspark.sql.functions import when
    weight_expr = None
    for label, weight in class_weights.items():
        if weight_expr is None:
            weight_expr = (when(col("Attack_type_index") == label, weight))
        else:
            weight_expr = weight_expr.when(col("Attack_type_index") == label, weight)
    df = df.withColumn("class_weight", weight_expr)

    # 6. FEATURE SELECTION: Select top 50 important features
    top_features = select_top_important_features(df, feature_cols, "Attack_type_index", num_features=50)


    return df, top_features

def modeling_proces(train_df, test_df, pipe_stages, features_col, num_trees, max_depth, seed):
    try:
        # INITIALIZE EVALUATION: Evaluator for weighted F1-score
        evaluator = MulticlassClassificationEvaluator(
            labelCol="Attack_type_index",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        # CLASSIFER INITIALIZATION: Initialize the Random Forest Classifier
        rf = RandomForestClassifier(
                labelCol="Attack_type_index",
                featuresCol= features_col,
                weightCol="class_weight",  # use the per-row weights!
                numTrees=num_trees,
                maxDepth=max_depth,
                seed=seed
            )
        # EVALUATION: Fit the model and evaluate on test data
        pipe_stages.append(rf)
        pipeline = Pipeline(stages=pipe_stages)
        model = pipeline.fit(train_df)
        test_pred = model.transform(test_df)
        train_pred = model.transform(train_df)
        test_score = evaluator.evaluate(test_pred)
        train_score = evaluator.evaluate(train_pred)
        
        return model, test_score, train_score
    
    except Exception as e:
            logger.error(f"Error applying Random Forest with PCA: {e}")

def modeling(df, feature_cols, seed):
    """
    Preprocess data and apply RandomForest with normalization and PCA.
    """
    num_trees = 35
    max_depth = 10
    pca_k = 10
    models = {}
    training_scores = {}
    testing_scores = {}
    
    try:
        logger.info("Class distribution of Attack_type:")
        df.groupBy("Attack_type").count().show()
        
        # Assemble final features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df)

        # DATA SPLITING: split dataset into training and testing 
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
        logger.info(f"Training set size: {train_df.count()}")
        logger.info(f"Test set size: {test_df.count()}")
        
        # MODEL 1: --------------------------------------------------------------
        # Model Only (Random Forest)
        model_1, training_score, testing_score = modeling_proces(train_df, test_df, [], "features",
                                                         num_trees, max_depth, seed)
        models["without_normalization"] = model_1
        training_scores["without_normalization"] = training_score
        testing_scores["without_normalization"] = testing_score

        logger.info("-" * 50)
        logger.info(f"Model 1 Train Score(f1): {round(training_score,4)}")
        logger.info(f"Model 1 Test Score(f1): {round(testing_score,4)}")
        logger.info("-" * 50)

        # MODEL 2: ---------------------------------------------------------------
        # Random Forest Classifier with Normalizarion
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

        model_2, training_score, testing_score = modeling_proces(train_df, test_df, [scaler], "scaled_features",
                                                         num_trees, max_depth, seed)
        models["with_normalization"] = model_2
        training_scores["with_normalization"] = training_score
        testing_scores["with_normalization"] = testing_score

        logger.info(f"Model 2 Train Score(f1): {round(training_score,4)}")
        logger.info(f"Model 2 Test Score(f1): {round(testing_score,4)}")
        logger.info("-" * 50)

        # MODEL 3: -----------------------------------------------------------------
        # Random Forest Classifier with PCA
        # Adjust PCA if needed
        sample = df.select("features").head()
        num_features = len(sample["features"])
        if num_features < pca_k:
            logger.warning(f"Reducing PCA components to {num_features}")
            pca_k = num_features

        pca = PCA(k=pca_k, inputCol="scaled_features", outputCol="pca_features")

        model_3, training_score, testing_score = modeling_proces(train_df, test_df, [scaler, pca], "pca_features",
                                                         num_trees, max_depth, seed)
        models["with_normalization_pca"] = model_3
        training_scores["with_normalization"] = training_score
        testing_scores["with_normalization"] = testing_score

        logger.info(f"Model 3 Train Score(f1): {round(training_score,4)}")
        logger.info(f"Model 3 Test Score(f1): {round(testing_score,4)}")
        logger.info("-" * 50)

        # PCA explained variance
        pca_model = model_3.stages[1]
        explained_variance = pca_model.explainedVariance.toArray()
        logger.info(f"PCA explained variance ratios: {explained_variance.tolist()}")

        logger.info("-" * 50)
        # ----------------------------------------------------------
        #logger.info(f"Train Weighted F1-score: {train_score}")
        #logger.info(f"Test Weighted F1-score: {test_score}")

        return training_scores, testing_scores

    except Exception as e:
        logger.error(f"Error during modeling: {e}")
        return None, 0.0

def main(spark):
    # Set data file
    file_name = "RT_IOT2022_normalized.csv"
    script_dir = Path(__file__).parent
    file_path = script_dir / file_name

    # Load data
    df = load_data(file_path, spark)

    # Data Preparation
    df, feature_cols = data_prepration(df)

    # Model Processing
    seed = 42
    training_scores, testing_scores = modeling(df, feature_cols, seed)

    #logger.info(f"Final test weighted F1-score: {test_score}")

if __name__ == "__main__":
    spark_session = SparkSession.builder.appName("IoT_Attack_Classification").getOrCreate()
    spark_session.sparkContext.setLogLevel("ERROR")
    try:
        main(spark_session)
    finally:
        spark_session.stop()
        print("Spark session stopped successfully!")
