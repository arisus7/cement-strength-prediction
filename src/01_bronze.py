# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Bronze: raw ingestion
# MAGIC
# MAGIC Loads `datathon_data_gcc.csv` into Unity Catalog **exactly as it arrives**: every column stays a string, so no value
# MAGIC is lost or coerced before Silver decides how to handle it. The only additions are lineage columns:
# MAGIC
# MAGIC | Column | Meaning |
# MAGIC |---|---|
# MAGIC | `_source_file` | Path of the file that was read |
# MAGIC | `_source_row` | 1-based line position in the source file (header excluded) |
# MAGIC | `_ingested_at` | Timestamp of this load |
# MAGIC
# MAGIC **Before the first run:** upload the CSV to the volume `<catalog>.<schema>.raw` (this notebook creates the volume).

# COMMAND ----------

dbutils.widgets.text("catalog", "datathon", "Catalog")
dbutils.widgets.text("schema", "gcc", "Schema")
dbutils.widgets.text("source_file", "datathon_data_gcc.csv", "Source file in raw volume")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SOURCE_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw/{dbutils.widgets.get('source_file')}"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_cement_quality"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")
print(f"Reading {SOURCE_PATH} -> {BRONZE_TABLE}")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Every column is read as string on purpose: typing is a Silver decision.
raw = (
    spark.read.format("csv")
    .option("header", True)
    .option("inferSchema", False)
    .option("mode", "FAILFAST")
    .load(SOURCE_PATH)
    .select("*", F.col("_metadata.file_path").alias("_source_file"))
)

EXPECTED_COLUMNS = [
    "ProductType",
    "Physical_Metric_A", "Physical_Metric_B",
    *[f"Mineral_Phase_{c}" for c in "ABCDEFGH"],
    "Chemical_Composition_A", "Chemical_Composition_B",
    "Strength1", "Strength7", "Strength28",
]
missing = set(EXPECTED_COLUMNS) - set(raw.columns)
assert not missing, f"Source file is missing columns: {sorted(missing)}"

# A single small CSV is read in order, so monotonically_increasing_id preserves file order.
bronze = (
    raw.withColumn("_mono_id", F.monotonically_increasing_id())
    .withColumn("_source_row", F.row_number().over(Window.orderBy("_mono_id")))
    .drop("_mono_id")
    .withColumn("_ingested_at", F.current_timestamp())
)

# COMMAND ----------

(
    bronze.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(BRONZE_TABLE)
)

spark.sql(f"COMMENT ON TABLE {BRONZE_TABLE} IS 'Raw cement quality samples, all columns as string. Source: datathon_data_gcc.csv'")

n = spark.table(BRONZE_TABLE).count()
print(f"Bronze rows: {n}")
assert n > 0, "Bronze table is empty"
display(spark.table(BRONZE_TABLE))