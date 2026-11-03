# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Gold: model-ready features
# MAGIC
# MAGIC Turns Silver into the table the EDA, the model and the dashboard all read from.
# MAGIC
# MAGIC | Step | What it does |
# MAGIC |---|---|
# MAGIC | G01 | Group rare product types (< `min_type_rows`) into `Other` and one-hot encode |
# MAGIC | G02 | Deterministic train/test split by `leakage_group`, so batch-copied rows never straddle the split |
# MAGIC | G03 | Median imputation of remaining missing inputs, **with medians from the train split only**, plus `_missing` flags |
# MAGIC | G04 | Write `gold_cement_features` and `gold_feature_spec` (feature list, roles and imputation values for serving) |
# MAGIC
# MAGIC Targets stay nullable: train one model per target on the rows where that target exists, or filter to rows with all
# MAGIC three for a joint multi-output model. Strength is in psi.

# COMMAND ----------

dbutils.widgets.text("catalog", "datathon", "Catalog")
dbutils.widgets.text("schema", "gcc", "Schema")
dbutils.widgets.text("test_pct", "20", "Test split %")
dbutils.widgets.text("split_salt", "micai2026", "Split salt")
dbutils.widgets.text("min_type_rows", "20", "Min rows to keep a ProductType")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TEST_PCT = int(dbutils.widgets.get("test_pct"))
SPLIT_SALT = dbutils.widgets.get("split_salt")
MIN_TYPE_ROWS = int(dbutils.widgets.get("min_type_rows"))

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_cement_quality"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.gold_cement_features"
SPEC_TABLE = f"{CATALOG}.{SCHEMA}.gold_feature_spec"

# COMMAND ----------

import re
from pyspark.sql import functions as F

PHASE_COLS = [f"Mineral_Phase_{c}" for c in "ABDEFGH"]
INPUT_COLS = ["Physical_Metric_A", "Physical_Metric_B", *PHASE_COLS, "Chemical_Composition_A", "Chemical_Composition_B"]
TARGET_COLS = ["Strength1", "Strength7", "Strength28"]

silver = spark.table(SILVER_TABLE)

# COMMAND ----------

# G01: rare product types -> Other, then one-hot
type_counts = {r["ProductType"]: r["count"] for r in silver.groupBy("ProductType").count().collect()}
kept_types = sorted(t for t, n in type_counts.items() if n >= MIN_TYPE_ROWS)
print(f"Kept types: {kept_types}; grouped into Other: { {t: n for t, n in type_counts.items() if t not in kept_types} }")

def type_col_name(t):
    return "pt_" + re.sub(r"\W+", "_", t.strip().lower())

df = silver.withColumn(
    "ProductType_group", F.when(F.col("ProductType").isin(kept_types), F.col("ProductType")).otherwise(F.lit("Other"))
)

# MODIFICACIÓN VIF: Excluimos "Other" de la lista de variables dummy a generar
type_groups = kept_types 
TYPE_COLS = [type_col_name(t) for t in type_groups]

for t, c in zip(type_groups, TYPE_COLS):
    df = df.withColumn(c, (F.col("ProductType_group") == t).cast("int"))

# COMMAND ----------

# G02: split on the leakage group, not the row, so rows that share a batch land on the same side
bucket = F.pmod(F.xxhash64(F.concat_ws("|", F.lit(SPLIT_SALT), F.col("leakage_group"))), F.lit(100))
df = df.withColumn("split", F.when(bucket < TEST_PCT, "test").otherwise("train"))

leaks = (
    df.groupBy("leakage_group").agg(F.countDistinct("split").alias("n_splits")).filter("n_splits > 1").count()
)
assert leaks == 0, f"{leaks} leakage groups span both splits"

display(df.groupBy("ProductType_group").pivot("split", ["train", "test"]).count().orderBy("ProductType_group"))

# COMMAND ----------

# G03: impute with train-only medians; add a missing flag wherever Silver still has nulls
train = df.filter("split = 'train'")
medians = train.select(*[F.percentile_approx(c, 0.5).alias(c) for c in INPUT_COLS]).first().asDict()
null_counts = df.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in INPUT_COLS]).first().asDict()

FLAG_COLS = []
for c in INPUT_COLS:
    if null_counts[c] and c != "Physical_Metric_A":
        flag = f"{c}_missing"
        df = df.withColumn(flag, F.col(c).isNull().cast("int"))
        
        # MODIFICACIÓN VIF: No agregar la bandera 'B' al arreglo final de features
        if flag != "Chemical_Composition_B_missing":
            FLAG_COLS.append(flag)
            
    df = df.withColumn(c, F.coalesce(F.col(c), F.lit(medians[c])))

df = df.withColumn("Physical_Metric_A_measured", F.col("Physical_Metric_A_measured").cast("int"))
FLAG_COLS = ["Physical_Metric_A_measured", *FLAG_COLS]

print("Train medians:", medians)
print("Missing before imputation:", {k: v for k, v in null_counts.items() if v})

# COMMAND ----------

# G04: write Gold + feature spec
FEATURE_COLS = [*INPUT_COLS, *FLAG_COLS, *TYPE_COLS]

gold = df.select(
    "sample_id",
    "split",
    "ProductType",
    "ProductType_group",
    *FEATURE_COLS,
    *TARGET_COLS,
    "leakage_group",
)

assert gold.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURE_COLS]).first().asDict() == {
    c: 0 for c in FEATURE_COLS
}, "Features still contain nulls"

(
    gold.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(GOLD_TABLE)
)
spark.sql(
    f"COMMENT ON TABLE {GOLD_TABLE} IS 'Model-ready features and psi strength targets. Split by leakage_group; "
    f"imputation from train medians. Feature list in gold_feature_spec.'"
)

spec_rows = (
    [(c, "feature", "numeric", float(medians[c])) for c in INPUT_COLS]
    + [(c, "feature", "flag", None) for c in FLAG_COLS]
    + [(c, "feature", "product_type", None) for c in TYPE_COLS]
    + [(c, "target", "strength_psi", None) for c in TARGET_COLS]
)
(
    spark.createDataFrame(spec_rows, "column string, role string, kind string, impute_value double")
    .withColumn("position", F.monotonically_increasing_id())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(SPEC_TABLE)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rows available per target and split

# COMMAND ----------

display(
    gold.groupBy("split").agg(
        F.count("*").alias("rows"),
        *[F.count(c).alias(f"{c}_rows") for c in TARGET_COLS],
        F.sum(F.when(F.col("Strength1").isNotNull() & F.col("Strength7").isNotNull() & F.col("Strength28").isNotNull(), 1)).alias("all_targets_rows"),
    )
)
print(f"Gold written: {GOLD_TABLE} ({gold.count()} rows, {len(FEATURE_COLS)} features)")