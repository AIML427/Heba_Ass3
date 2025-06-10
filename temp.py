from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, stddev
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Spark session with increased memory
spark = SparkSession.builder.appName("NetworkDataCleaning") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

def load_data(file_path, output_file_path):
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        spark_file_path = f"file://{file_path.as_posix()}"
        logger.info(f"Loading file: {spark_file_path}")

        df_raw = spark.read.option("delimiter", "\t").option("header", "true").csv(spark_file_path)
        
        columns = df_raw.columns
        logger.info(f"Detected columns ({len(columns)}): {columns}")
        
        if len(columns) <= 1:
            logger.warning("Only one column detected. Loading as raw text.")
            df_text = spark.read.text(spark_file_path)
            logger.info("Raw file content (first 5 rows):")
            df_text.show(5, truncate=False)
            return None, None

        normalized_columns = [c.replace('.', '_') for c in columns]
        df_raw = df_raw.toDF(*normalized_columns)
        logger.info(f"Normalized columns ({len(normalized_columns)}): {normalized_columns}")

        logger.info("Null counts and invalid value ('-') counts per column:")
        for col_name in normalized_columns:
            null_count = df_raw.filter(col(col_name).isNull()).count()
            invalid_count = df_raw.filter(col(col_name).isin("-", "nan", "null")).count()
            logger.info(f"{col_name}: {null_count} nulls, {invalid_count} invalid values")

        for col_name in normalized_columns:
            if col_name in ["proto", "service", "Attack_type"]:
                df_raw = df_raw.withColumn(col_name, when(col(col_name).isNotNull(), col(col_name).lower()).otherwise(col(col_name)))
            df_raw = df_raw.withColumn(col_name, when(col(col_name).isin("-", "nan", "null"), "0").otherwise(col(col_name)))

        logger.info("Sample of raw data (post-preprocessing):")
        df_raw.show(5, truncate=False)

        output_file_path = Path(output_file_path).resolve()
        spark_output_path = f"file://{output_file_path.as_posix()}"
        logger.info(f"Saving normalized data to: {spark_output_path}")
        df_raw.coalesce(1).write.option("delimiter", "\t").option("header", "true").mode("overwrite").csv(spark_output_path)

        logger.info(f"Verifying saved file: {spark_output_path}")
        df_verify = spark.read.option("delimiter", "\t").option("header", "true").csv(spark_output_path)
        logger.info(f"Saved file columns ({len(df_verify.columns)}): {df_verify.columns}")
        df_verify.show(5, truncate=False)

        return df_raw, normalized_columns
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, []

def clean_data(df, normalized_columns):
    try:
        columns = [
            "id_orig_p", "id_resp_p", "proto", "service", "flow_duration", "fwd_pkts_tot", 
            "bwd_pkts_tot", "fwd_data_pkts_tot", "bwd_data_pkts_tot", "fwd_pkts_per_sec", 
            "bwd_pkts_per_sec", "flow_pkts_per_sec", "down_up_ratio", "fwd_header_size_tot", 
            "fwd_header_size_min", "fwd_header_size_max", "bwd_header_size_tot", 
            "bwd_header_size_min", "bwd_header_size_max", "flow_FIN_flag_count", 
            "flow_SYN_flag_count", "flow_RST_flag_count", "fwd_PSH_flag_count", 
            "bwd_PSH_flag_count", "flow_ACK_flag_count", "fwd_URG_flag_count", 
            "bwd_URG_flag_count", "flow_CWR_flag_count", "flow_ECE_flag_count", 
            "fwd_pkts_payload_min", "fwd_pkts_payload_max", "fwd_pkts_payload_tot", 
            "fwd_pkts_payload_avg", "fwd_pkts_payload_std", "bwd_pkts_payload_min", 
            "bwd_pkts_payload_max", "bwd_pkts_payload_tot", "bwd_pkts_payload_avg", 
            "bwd_pkts_payload_std", "flow_pkts_payload_min", "flow_pkts_payload_max", 
            "flow_pkts_payload_tot", "flow_pkts_payload_avg", "flow_pkts_payload_std", 
            "fwd_iat_min", "fwd_iat_max", "fwd_iat_tot", "fwd_iat_avg", "fwd_iat_std", 
            "bwd_iat_min", "bwd_iat_max", "bwd_iat_tot", "bwd_iat_avg", "bwd_iat_std", 
            "flow_iat_min", "flow_iat_max", "flow_iat_tot", "flow_iat_avg", "flow_iat_std", 
            "payload_bytes_per_second", "fwd_subflow_pkts", "bwd_subflow_pkts", 
            "fwd_subflow_bytes", "bwd_subflow_bytes", "fwd_bulk_bytes", "bwd_bulk_bytes", 
            "fwd_bulk_packets", "bwd_bulk_packets", "fwd_bulk_rate", "bwd_bulk_rate", 
            "active_min", "active_max", "active_tot", "active_avg", "active_std", 
            "idle_min", "idle_max", "idle_tot", "idle_avg", "idle_std", 
            "fwd_init_window_size", "bwd_init_window_size", "fwd_last_window_size", 
            "Attack_type"
        ]

        string_cols = {"proto", "service", "Attack_type"}
        integer_cols = {
            "id_orig_p", "id_resp_p", "fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot", 
            "bwd_data_pkts_tot", "fwd_header_size_tot", "fwd_header_size_min", 
            "fwd_header_size_max", "bwd_header_size_tot", "bwd_header_size_min", 
            "bwd_header_size_max", "flow_FIN_flag_count", "flow_SYN_flag_count", 
            "flow_RST_flag_count", "fwd_PSH_flag_count", "bwd_PSH_flag_count", 
            "flow_ACK_flag_count", "fwd_URG_flag_count", "bwd_URG_flag_count", 
            "flow_CWR_flag_count", "flow_ECE_flag_count", "fwd_init_window_size", 
            "bwd_init_window_size", "fwd_last_window_size"
        }

        schema = StructType([
            StructField(name, 
                       StringType() if name in string_cols else 
                       IntegerType() if name in integer_cols else 
                       DoubleType(), 
                       True)
            for name in columns
        ])

        if len(normalized_columns) != len(schema.fields):
            logger.error(f"Column count mismatch: Expected {len(schema.fields)}, Found {len(normalized_columns)}")
            return None

        for field in schema.fields:
            df = df.withColumn(field.name, col(field.name).cast(field.dataType))

        # Check non-numeric values for all numerical columns
        logger.info("Checking for non-numeric values in numerical columns")
        numerical_cols = [f.name for f in schema.fields if f.dataType in [DoubleType(), IntegerType()]]
        for col_name in numerical_cols:
            non_numeric = df.filter(~col(col_name).cast(f.dataType).isNotNull() & col(col_name).isNotNull())
            count = non_numeric.count()
            logger.info(f"{col_name}: {count} non-numeric values")
            if count > 0:
                logger.info(f"Sample non-numeric values for {col_name}:")
                non_numeric.select(col_name).show(5, truncate=False)

        # Impute nulls for all numerical columns
        numerical_cols += ["proto_index", "service_index"]
        logger.info(f"Imputing nulls for numerical columns: {numerical_cols}")
        try:
            imputer = Imputer(
                inputCols=numerical_cols,
                outputCols=numerical_cols,
                strategy="median"
            )
            df = imputer.fit(df).transform(df)
        except Exception as e:
            logger.error(f"Imputer failed: {e}")
            logger.info("Falling back to default value (0)")
            for col_name in numerical_cols:
                default_val = 0.0 if col_name != "fwd_pkts_tot" else 0
                df = df.withColumn(col_name, when(col(col_name).isNull(), default_val).otherwise(col(col_name)))

        # Replace NaNs and infinities
        for col_name in numerical_cols:
            df = df.withColumn(col_name, when(col(col_name).isNull() | isnan(col_name) | col(col_name).isin(float('inf'), float('-inf')), 0.0).otherwise(col(col_name)))

        # Handle nulls in categorical columns
        df = df.withColumn("proto", when(col("proto").isNull(), "unknown").otherwise(col("proto")))
        df = df.withColumn("service", when(col("service").isNull(), "unknown").otherwise(col("service")))
        df = df.withColumn("Attack_type", when(col("Attack_type").isNull(), "unknown").otherwise(col("Attack_type")))

        # Encode categorical features
        logger.info("Encoding categorical features: proto, service, Attack_type")
        proto_indexer = StringIndexer(inputCol="proto", outputCol="proto_index", handleInvalid="keep")
        service_indexer = StringIndexer(inputCol="service", outputCol="service_index", handleInvalid="keep")
        attack_indexer = StringIndexer(inputCol="Attack_type", outputCol="Attack_type_index", handleInvalid="keep")
        proto_encoder = OneHotEncoder(inputCols=["proto_index"], outputCols=["proto_vec"])
        service_encoder = OneHotEncoder(inputCols=["service_index"], outputCols=["service_vec"])

        df = proto_indexer.fit(df).transform(df)
        df = service_indexer.fit(df).transform(df)
        df = attack_indexer.fit(df).transform(df)
        df = proto_encoder.fit(df).transform(df)
        df = service_encoder.fit(df).transform(df)

        logger.info("Sample of encoded categorical features:")
        df.select("proto", "proto_index", "proto_vec", "service", "service_index", "service_vec", "Attack_type", "Attack_type_index").show(5, truncate=False)

        # Check for constant columns
        logger.info("Checking for constant numerical columns")
        for col_name in numerical_cols:
            std = df.agg(stddev(col_name).alias("std")).collect()[0]["std"]
            if std == 0 or std is None:
                logger.warning(f"{col_name} has zero or null standard deviation, replacing with 0")
                df = df.withColumn(col_name, when(col(col_name).isNotNull(), 0.0).otherwise(col(col_name)))

        # Cleaning Steps
        df = df.dropDuplicates()

        for col_name in ["flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "fwd_data_pkts_tot", "bwd_data_pkts_tot"]:
            df = df.withColumn(col_name, when(col(col_name) < 0, 0).otherwise(col(col_name)))

        logger.info("Null counts in cleaned dataset:")
        for col_name in df.columns:
            null_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
            logger.info(f"{col_name}: {null_count} nulls")

        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        return None

def apply_random_forest(df, num_trees=50, max_depth=5, seed=42, pca_k=10):
    try:
        logger.info("Applying Random Forest in three configurations: (1) Model Only, (2) With Normalization, (3) With Normalization and PCA")

        logger.info("Class distribution of Attack_type_index:")
        df.groupBy("Attack_type_index").count().show()

        feature_cols = [c for c in df.columns if c not in ["Attack_type_index", "proto", "service", "Attack_type"]]
        logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

        train_df, test_df = df.randomSplit([0.8, 0.2], seed=seed)
        logger.info(f"Training set size: {train_df.count()} rows")
        logger.info(f"Test set size: {test_df.count()} rows")

        evaluator = MulticlassClassificationEvaluator(
            labelCol="Attack_type_index",
            predictionCol="prediction",
            metricName="weightedF1"
        )

        models = {}
        f1_scores = {}

        # Run 1: Model Only
        try:
            logger.info("Run 1: Training Random Forest without normalization or PCA")
            rf1 = RandomForestClassifier(
                labelCol="Attack_type_index",
                featuresCol="features",
                numTrees=num_trees,
                maxDepth=max_depth,
                seed=seed,
                weightCol=None
            )
            pipeline1 = Pipeline(stages=[assembler, rf1])
            model1 = pipeline1.fit(train_df)
            predictions1 = model1.transform(test_df)
            f1_score1 = evaluator.evaluate(predictions1)
            logger.info(f"Run 1 Weighted F1-score: {f1_score1}")
            logger.info("Run 1 Sample predictions:")
            predictions1.select("Attack_type_index", "prediction", "probability").show(5, truncate=False)
            rf_model1 = model1.stages[1]
            feature_importance1 = rf_model1.featureImportances.toArray()
            importance_pairs1 = [(feature_cols[i], feature_importance1[i]) for i in range(len(feature_cols))]
            importance_pairs1.sort(key=lambda x: x[1], reverse=True)
            logger.info("Run 1 Top 10 feature importances:")
            for feature, importance in importance_pairs1[:10]:
                logger.info(f"{feature}: {importance}")
            models["model_only"] = model1
            f1_scores["model_only"] = f1_score1
        except Exception as e:
            logger.error(f"Run 1 failed: {e}")
            models["model_only"] = None
            f1_scores["model_only"] = 0.0

        # Run 2: Model with Normalization
        try:
            logger.info("Run 2: Training Random Forest with normalization")
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
            rf2 = RandomForestClassifier(
                labelCol="Attack_type_index",
                featuresCol="scaled_features",
                numTrees=num_trees,
                maxDepth=max_depth,
                seed=seed,
                weightCol=None
            )
            pipeline2 = Pipeline(stages=[assembler, scaler, rf2])
            model2 = pipeline2.fit(train_df)
            predictions2 = model2.transform(test_df)
            f1_score2 = evaluator.evaluate(predictions2)
            logger.info(f"Run 2 Weighted F1-score: {f1_score2}")
            logger.info("Run 2 Sample predictions:")
            predictions2.select("Attack_type_index", "prediction", "probability").show(5, truncate=False)
            rf_model2 = model2.stages[2]
            feature_importance2 = rf_model2.featureImportances.toArray()
            importance_pairs2 = [(feature_cols[i], feature_importance2[i]) for i in range(len(feature_cols))]
            importance_pairs2.sort(key=lambda x: x[1], reverse=True)
            logger.info("Run 2 Top 10 feature importances:")
            for feature, importance in importance_pairs2[:10]:
                logger.info(f"{feature}: {importance}")
            models["with_normalization"] = model2
            f1_scores["with_normalization"] = f1_score2
        except Exception as e:
            logger.error(f"Run 2 failed: {e}")
            models["with_normalization"] = None
            f1_scores["with_normalization"] = 0.0

        # Run 3: Model with Normalization and PCA
        try:
            logger.info("Run 3: Training Random Forest with normalization and PCA")
            pca = PCA(k=pca_k, inputCol="scaled_features", outputCol="pca_features")
            rf3 = RandomForestClassifier(
                labelCol="Attack_type_index",
                featuresCol="pca_features",
                numTrees=num_trees,
                maxDepth=max_depth,
                seed=seed,
                weightCol=None
            )
            pipeline3 = Pipeline(stages=[assembler, scaler, pca, rf3])
            model3 = pipeline3.fit(train_df)
            pca_model = model3.stages[2]
            explained_variance = pca_model.explainedVariance.toArray()
            cumulative_variance = np.cumsum(explained_variance)
            logger.info(f"Run 3 Explained variance ratios for {pca_k} components: {explained_variance.tolist()}")
            logger.info(f"Run 3 Cumulative variance explained: {cumulative_variance[-1]:.4f}")
            predictions3 = model3.transform(test_df)
            f1_score3 = evaluator.evaluate(predictions3)
            logger.info(f"Run 3 Weighted F1-score: {f1_score3}")
            logger.info("Run 3 Sample predictions:")
            predictions3.select("Attack_type_index", "prediction", "probability").show(5, truncate=False)
            rf_model3 = model3.stages[3]
            feature_importance3 = rf_model3.featureImportances.toArray()
            pca_loadings = pca_model.pc.toArray()
            original_feature_importance3 = np.abs(pca_loadings.T).dot(feature_importance3)
            importance_pairs3 = [(feature_cols[i], original_feature_importance3[i]) for i in range(len(feature_cols))]
            importance_pairs3.sort(key=lambda x: x[1], reverse=True)
            logger.info("Run 3 Top 10 original feature importances (via PCA):")
            for feature, importance in importance_pairs3[:10]:
                logger.info(f"{feature}: {importance}")
            models["with_normalization_pca"] = model3
            f1_scores["with_normalization_pca"] = f1_score3
        except Exception as e:
            logger.error(f"Run 3 failed: {e}")
            models["with_normalization_pca"] = None
            f1_scores["with_normalization_pca"] = 0.0

        logger.info("Summary of Weighted F1-scores:")
        logger.info(f"Run 1 (Model Only): {f1_scores['model_only']}")
        logger.info(f"Run 2 (With Normalization): {f1_scores['with_normalization']}")
        logger.info(f"Run 3 (With Normalization and PCA): {f1_scores['with_normalization_pca']}")

        return models, f1_scores
    except Exception as e:
        logger.error(f"Error applying Random Forest: {e}")
        return {}, {}

def process_data(file_path, output_file_path="data_samples_normalized.data"):
    try:
        df, normalized_columns = load_data(file_path, output_file_path)
        if df is None or not normalized_columns:
            logger.error("Data loading failed.")
            return None

        df_cleaned = clean_data(df, normalized_columns)
        if df_cleaned is None:
            logger.error("Data cleaning failed.")
            return None

        models, f1_scores = apply_random_forest(df_cleaned)
        if not models:
            logger.error("Model training failed.")
            return None

        return df_cleaned, models, f1_scores
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None
    finally:
        spark.stop()

# Example usage
file_path = "data_samples.data"  # Replace with your actual file path
output_file_path = "data_samples_normalized.data"
result = process_data(file_path, output_file_path)

if result is not None:
    df_cleaned, models, f1_scores = result
    df_cleaned.printSchema()
    df_cleaned.show(5, truncate=False)
    print("Random Forest Weighted F1-scores:")
    print(f"Model Only: {f1_scores.get('model_only', 'N/A')}")
    print(f"With Normalization: {f1_scores.get('with_normalization', 'N/A')}")
    print(f"With Normalization and PCA: {f1_scores.get('with_normalization_pca', 'N/A')}")
else:
    print("Data processing failed. Check logs for details.")