# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: cleaned, typed, validated
# MAGIC
# MAGIC Reads Bronze and applies the cleaning rules below **in order**. Every rule records how many rows it touched in
# MAGIC `silver_dq_log`, so each run is traceable.
# MAGIC
# MAGIC | Rule | What it does | Why |
# MAGIC |---|---|---|
# MAGIC | S01 | Trim text, map null tokens (`''`, `NA`, `nan`, `null`) to null, cast inputs/targets to double | Bronze is all strings |
# MAGIC | S02 | Drop exact duplicate rows | 5 exact copies in the source |
# MAGIC | S03 | Drop rows where all 8 mineral phases are null | The phases (Bogue) drive hydration and strength; imputing all 8 would invent the sample |
# MAGIC | S04 | Drop rows with no strength target at all | Nothing to learn from |
# MAGIC | S05 | Null out scaled inputs outside [0, 1] | Source inputs are min-max scaled; anything outside is corrupt |
# MAGIC | S06 | Null out strengths outside a plausible psi range | Guards against unit or typing errors |
# MAGIC | S07 | Drop rows where R1 ≤ R7 ≤ R28 does not hold | Strength cannot decrease with curing age |
# MAGIC | S08 | Flag `Physical_Metric_A` as measured or not | 76% missing: the fact it was measured is the usable signal |
# MAGIC | S09 | Flag rows that share a strength triplet with other rows, or whose inputs map to conflicting targets | Likely one batch's results copied across samples; Gold keeps them in the same split to avoid leakage |
# MAGIC
# MAGIC Remaining missing inputs (`Physical_Metric_B`, `Chemical_Composition_*`, one `Mineral_Phase_E`) are **not**
# MAGIC imputed here: imputation statistics must come from the training split only, so it happens in Gold.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "cement_strength", "Schema")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_cement_quality"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_cement_quality"
DQ_LOG_TABLE = f"{CATALOG}.{SCHEMA}.silver_dq_log"

# COMMAND ----------

import uuid
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PHASE_COLS = [f"Mineral_Phase_{c}" for c in "ABCDEFGH"]
SCALED_COLS = ["Physical_Metric_A", *PHASE_COLS, "Chemical_Composition_A", "Chemical_Composition_B"]
# Physical_Metric_B looks like a z-score (range about -2.5 to 8.6), so it is not range-checked against [0, 1].
INPUT_COLS = ["Physical_Metric_A", "Physical_Metric_B", *PHASE_COLS, "Chemical_Composition_A", "Chemical_Composition_B"]
TARGET_COLS = ["Strength1", "Strength7", "Strength28"]
NUMERIC_COLS = INPUT_COLS + TARGET_COLS
BUSINESS_COLS = ["ProductType", *NUMERIC_COLS]

# Strength in psi: source spans 1,600-9,130. Bounds are deliberately loose; they only catch gross errors.
STRENGTH_PSI_MIN, STRENGTH_PSI_MAX = 500.0, 15000.0
NULL_TOKENS = ["", "na", "n/a", "nan", "null", "none"]

RUN_ID = str(uuid.uuid4())
dq_log = []


def log_rule(rule_id, description, rows_before, rows_after, rows_affected=None):
    affected = rows_before - rows_after if rows_affected is None else rows_affected
    dq_log.append((RUN_ID, rule_id, description, rows_before, rows_after, affected))
    print(f"{rule_id}: {description} -> affected {affected} (rows {rows_before} -> {rows_after})")

# COMMAND ----------

# S01: normalise text and cast
bronze = spark.table(BRONZE_TABLE)
n0 = bronze.count()


def clean_str(col):
    trimmed = F.trim(F.col(col))
    return F.when(trimmed.isNull() | F.lower(trimmed).isin(NULL_TOKENS), None).otherwise(trimmed)


df = bronze.select(
    clean_str("ProductType").alias("ProductType"),
    *[clean_str(c).alias(c) for c in NUMERIC_COLS],
    "_source_file", "_source_row", "_ingested_at",
)

# Values that are present but fail to parse are counted before they become null.
cast_failures = df.select(
    *[F.sum((F.col(c).isNotNull() & F.col(c).cast("double").isNull()).cast("int")).alias(c) for c in NUMERIC_COLS]
).first().asDict()
df = df.select("ProductType", *[F.col(c).cast("double").alias(c) for c in NUMERIC_COLS], "_source_file", "_source_row", "_ingested_at")
log_rule("S01", f"Cast to double; unparseable values nulled: { {k: v for k, v in cast_failures.items() if v} }",
         n0, n0, sum(v or 0 for v in cast_failures.values()))

null_product = df.filter(F.col("ProductType").isNull()).count()
assert null_product == 0, f"{null_product} rows have no ProductType"

# COMMAND ----------

# S02: exact duplicates (first occurrence in file order is kept)
w = Window.partitionBy(*BUSINESS_COLS).orderBy("_source_row")
n_before = df.count()
df = df.withColumn("_dup_rank", F.row_number().over(w)).filter("_dup_rank = 1").drop("_dup_rank")
log_rule("S02", "Drop exact duplicate rows", n_before, df.count())

# S03: no mineral phase data
n_before = df.count()
df = df.filter(~reduce(lambda a, b: a & b, [F.col(c).isNull() for c in PHASE_COLS]))
log_rule("S03", "Drop rows with all 8 mineral phases null", n_before, df.count())

# S04: no target at all
n_before = df.count()
df = df.filter(reduce(lambda a, b: a | b, [F.col(c).isNotNull() for c in TARGET_COLS]))
log_rule("S04", "Drop rows with no strength target", n_before, df.count())

# COMMAND ----------

# S05: scaled inputs outside [0, 1]
n_now = df.count()
out_of_range = df.select(
    *[F.sum((~F.col(c).between(0.0, 1.0)).cast("int")).alias(c) for c in SCALED_COLS]
).first().asDict()
for c in SCALED_COLS:
    df = df.withColumn(c, F.when(F.col(c).between(0.0, 1.0), F.col(c)))
log_rule("S05", f"Null scaled inputs outside [0,1]: { {k: v for k, v in out_of_range.items() if v} }",
         n_now, n_now, sum(v or 0 for v in out_of_range.values()))

# S06: implausible strengths
implausible = df.select(
    *[F.sum((~F.col(c).between(STRENGTH_PSI_MIN, STRENGTH_PSI_MAX)).cast("int")).alias(c) for c in TARGET_COLS]
).first().asDict()
for c in TARGET_COLS:
    df = df.withColumn(c, F.when(F.col(c).between(STRENGTH_PSI_MIN, STRENGTH_PSI_MAX), F.col(c)))
log_rule("S06", f"Null strengths outside [{STRENGTH_PSI_MIN}, {STRENGTH_PSI_MAX}] psi: { {k: v for k, v in implausible.items() if v} }",
         n_now, n_now, sum(v or 0 for v in implausible.values()))

# S07: strength must not decrease with age (checked on whichever pairs are present)
order_ok = (
    (F.col("Strength1").isNull() | F.col("Strength7").isNull() | (F.col("Strength1") <= F.col("Strength7")))
    & (F.col("Strength7").isNull() | F.col("Strength28").isNull() | (F.col("Strength7") <= F.col("Strength28")))
    & (F.col("Strength1").isNull() | F.col("Strength28").isNull() | (F.col("Strength1") <= F.col("Strength28")))
)
n_before = df.count()
df = df.filter(order_ok)
log_rule("S07", "Drop rows where strength decreases with age", n_before, df.count())

# COMMAND ----------

# S08: measurement flag for the mostly-empty physical metric
df = df.withColumn("Physical_Metric_A_measured", F.col("Physical_Metric_A").isNotNull())
n_now = df.count()
log_rule("S08", "Flag Physical_Metric_A measured", n_now, n_now, df.filter("Physical_Metric_A_measured").count())

# S09: leakage groups.
# Rows with an identical, complete strength triplet probably come from one batch; rows with identical inputs but
# different strengths are conflicting labels. Both get a shared group id so Gold never splits them across train/test.
triplet_key = F.when(
    reduce(lambda a, b: a & b, [F.col(c).isNotNull() for c in TARGET_COLS]),
    F.concat_ws("|", *[F.col(c).cast("string") for c in TARGET_COLS]),
)
input_key = F.concat_ws("|", "ProductType", *[F.coalesce(F.col(c).cast("string"), F.lit("∅")) for c in INPUT_COLS])

df = (
    df.withColumn("sample_id", F.sha2(F.concat_ws("|", "ProductType", *[F.coalesce(F.col(c).cast("string"), F.lit("∅")) for c in NUMERIC_COLS]), 256))
    .withColumn("_triplet_key", triplet_key)
    .withColumn("_input_key", input_key)
    .withColumn("_triplet_n", F.when(F.col("_triplet_key").isNotNull(), F.count("*").over(Window.partitionBy("_triplet_key"))).otherwise(1))
    .withColumn("_input_n", F.count("*").over(Window.partitionBy("_input_key")))
    .withColumn("shared_strength_triplet", F.col("_triplet_n") > 1)
    .withColumn("conflicting_inputs", F.col("_input_n") > 1)
)

# Connect the two kinds of link (triplet and identical inputs) with a small iterative label propagation,
# so a chain of shared links ends up in one group.
df = df.withColumn("leakage_group", F.col("sample_id"))
for _ in range(5):
    by_triplet = Window.partitionBy(F.coalesce("_triplet_key", "sample_id"))
    by_input = Window.partitionBy("_input_key")
    df = df.withColumn("leakage_group", F.min("leakage_group").over(by_triplet))
    df = df.withColumn("leakage_group", F.min("leakage_group").over(by_input))

grouped_rows = df.filter(F.col("shared_strength_triplet") | F.col("conflicting_inputs")).count()
log_rule("S09", "Flag rows sharing strength triplet or conflicting inputs (leakage_group)", n_now, n_now, grouped_rows)

# COMMAND ----------

silver = df.select(
    "sample_id",
    "ProductType",
    *INPUT_COLS,
    "Physical_Metric_A_measured",
    *TARGET_COLS,
    "shared_strength_triplet",
    "conflicting_inputs",
    "leakage_group",
    "_source_file",
    "_source_row",
    "_ingested_at",
)

assert silver.select("sample_id").distinct().count() == silver.count(), "sample_id is not unique after dedup"

(
    silver.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(SILVER_TABLE)
)
spark.sql(f"COMMENT ON TABLE {SILVER_TABLE} IS 'Cleaned cement quality samples. Strength in psi. Rules S01-S09, see silver_dq_log.'")

dq_df = spark.createDataFrame(
    dq_log, "run_id string, rule_id string, description string, rows_before long, rows_after long, rows_affected long"
).withColumn("run_at", F.current_timestamp())
dq_df.write.format("delta").mode("append").saveAsTable(DQ_LOG_TABLE)

print(f"Silver rows: {silver.count()} (from {n0} in Bronze)")
display(dq_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remaining missing values (handled in Gold)

# COMMAND ----------

display(silver.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in INPUT_COLS + TARGET_COLS]))
display(silver.groupBy("ProductType").count().orderBy("ProductType"))
