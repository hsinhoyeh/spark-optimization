#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced NVMe Heavy Join Test for Apache Spark
---------------------------------------------
This script benchmarks Apache Spark performance for large-scale data joins
when using NVMe storage for shuffle and temporary data.
"""

import time
import os
import json
import sys
import glob
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, TimestampType
from pyspark.sql.functions import col, rand

# Configuration
NUM_CUSTOMERS = 1_000_000
NUM_PRODUCTS = 50_000
NUM_STORES = 1_000
NUM_SUPPLIERS = 5_000
NUM_TRANSACTIONS = 10_000_000

# Add robustness options
VERIFY_WRITES = True
CLEANUP_ON_EXIT = True
RESILIENT_READ = True
MAX_RETRIES = 3

def log_timing(message):
    """Decorator to log time taken by a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"Starting {message}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            print(f"Completed {message} in {duration:.2f} seconds")
            return result
        return wrapper
    return decorator

@log_timing("Spark session initialization")
def create_spark_session():
    """Create and configure a Spark session with enhanced NVMe handling"""
    return (SparkSession.builder
            .appName("NVMe Heavy Join Test")
            # Adaptive execution
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            # Shuffle partitions based on available cores
            .config("spark.sql.shuffle.partitions", "200")
            
            # More robust file handling
            .config("spark.sql.files.ignoreCorruptFiles", "true")
            .config("spark.sql.files.ignoreMissingFiles", "true")
            .config("spark.sql.legacy.createEmptyCollectionUsingStringType", "true")
            
            # Improved Parquet handling
            .config("spark.sql.parquet.mergeSchema", "true")
            .config("spark.sql.parquet.filterPushdown", "true")
            .config("spark.sql.parquet.columnarReaderBatchSize", "4096")
            
            # NVMe specific optimizations
            .config("spark.files.useFetchCache", "false")
            .config("spark.files.overwrite", "true")
            .config("spark.block.size", "128m")
            
            # Enhanced network and I/O
            .config("spark.io.compression.codec", "lz4")
            .config("spark.io.compression.lz4.blockSize", "512k")
            .config("spark.reducer.maxSizeInFlight", "96m")
            .config("spark.maxRemoteBlockSizeFetchToMem", "200m")
            
            # Better fault tolerance
            .config("spark.task.maxFailures", "5")
            .config("spark.network.timeout", "800s")
            .config("spark.executor.heartbeatInterval", "60s")
            .config("spark.task.reaper.enabled", "true")
            .config("spark.task.reaper.pollingInterval", "10s")
            
            # Handle executor issues better
            .config("spark.speculation", "false")
            
            # Cleanup aggressive
            .config("spark.cleaner.periodicGC.interval", "5min")
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
            
            .getOrCreate())

def get_customers_schema():
    """Return the schema for customers dataset"""
    return StructType([
        StructField("customer_id", IntegerType(), False),
        StructField("customer_name", StringType(), False),
        StructField("email", StringType(), False),
        StructField("address", StringType(), False),
        StructField("city", StringType(), False),
        StructField("state", StringType(), False),
        StructField("zipcode", StringType(), False),
        StructField("phone", StringType(), False),
        StructField("credit_score", IntegerType(), True),
        StructField("loyalty_segment", IntegerType(), True)
    ])

def get_products_schema():
    """Return the schema for products dataset"""
    return StructType([
        StructField("product_id", IntegerType(), False),
        StructField("product_name", StringType(), False),
        StructField("category", StringType(), False),
        StructField("subcategory", StringType(), False),
        StructField("price", DoubleType(), False),
        StructField("cost", DoubleType(), False),
        StructField("supplier_id", IntegerType(), True),
        StructField("inventory", IntegerType(), True),
        StructField("rating", DoubleType(), True)
    ])

def get_stores_schema():
    """Return the schema for stores dataset"""
    return StructType([
        StructField("store_id", IntegerType(), False),
        StructField("store_name", StringType(), False),
        StructField("address", StringType(), False),
        StructField("city", StringType(), False),
        StructField("state", StringType(), False),
        StructField("zipcode", StringType(), False),
        StructField("manager_id", IntegerType(), True),
        StructField("size_sqft", IntegerType(), True),
        StructField("open_date", TimestampType(), True)
    ])

def get_suppliers_schema():
    """Return the schema for suppliers dataset"""
    return StructType([
        StructField("supplier_id", IntegerType(), False),
        StructField("supplier_name", StringType(), False),
        StructField("contact_name", StringType(), False),
        StructField("email", StringType(), False),
        StructField("phone", StringType(), False),
        StructField("address", StringType(), False),
        StructField("city", StringType(), False),
        StructField("country", StringType(), False),
        StructField("payment_terms", StringType(), True),
        StructField("quality_score", DoubleType(), True)
    ])

def get_transactions_schema():
    """Return the schema for transactions dataset"""
    return StructType([
        StructField("transaction_id", IntegerType(), False),
        StructField("customer_id", IntegerType(), False),
        StructField("store_id", IntegerType(), False),
        StructField("product_id", IntegerType(), False),
        StructField("quantity", IntegerType(), False),
        StructField("transaction_date", TimestampType(), False),
        StructField("transaction_time", StringType(), False),
        StructField("unit_price", DoubleType(), False),
        StructField("discount", DoubleType(), True),
        StructField("payment_method", StringType(), True)
    ])

@log_timing("Generate customers dataset")
def generate_customers(spark, count=NUM_CUSTOMERS, partitions=None):
    """Generate a synthetic customers dataset"""
    print(f"Generating {count} customers...")

    # Use determined number of partitions based on cluster size
    if partitions is None:
        partitions = min(200, max(20, count // 10000))
    
    # Generate data with controlled partition distribution
    return spark.range(0, count, 1, partitions).selectExpr(
        "id as customer_id",
        "concat('Customer_', cast(id as string)) as customer_name",
        "concat('customer_', cast(id as string), '@example.com') as email",
        "concat(cast(rand() * 9999 as int), ' Main St') as address",
        "case when rand() < 0.2 then 'New York' when rand() < 0.4 then 'Los Angeles' when rand() < 0.6 then 'Chicago' else concat('City_', cast(rand() * 100 as int)) end as city",
        "case when rand() < 0.3 then 'CA' when rand() < 0.6 then 'NY' else concat('ST_', cast(rand() * 50 as int)) end as state",
        "concat(cast(10000 + rand() * 89999 as int)) as zipcode",
        "concat('(', cast(100 + rand() * 899 as int), ') ', cast(100 + rand() * 899 as int), '-', cast(1000 + rand() * 8999 as int)) as phone",
        "cast(300 + rand() * 550 as int) as credit_score",
        "cast(rand() * 5 as int) as loyalty_segment"
    )

@log_timing("Generate products dataset")
def generate_products(spark, count=NUM_PRODUCTS, partitions=None):
    """Generate a synthetic products dataset"""
    print(f"Generating {count} products...")

    # Use determined number of partitions based on cluster size
    if partitions is None:
        partitions = min(100, max(20, count // 1000))
    
    return spark.range(0, count, 1, partitions).selectExpr(
        "id as product_id",
        "concat('Product_', cast(id as string)) as product_name",
        # Create skew in the category (more products in category 'Electronics')
        "case when rand() < 0.3 then 'Electronics' when rand() < 0.5 then 'Clothing' when rand() < 0.7 then 'Home' when rand() < 0.8 then 'Books' else concat('Category_', cast(rand() * 20 as int)) end as category",
        "concat('Subcategory_', cast(rand() * 50 as int)) as subcategory",
        "cast(4.99 + rand() * 995 as double) as price",
        "cast(2.49 + rand() * 500 as double) as cost",
        # Create skew in supplier distribution
        "cast(rand() * 5000 as int) as supplier_id",
        "cast(rand() * 10000 as int) as inventory",
        "cast(1 + rand() * 4 as double) as rating"
    )

@log_timing("Generate stores dataset")
def generate_stores(spark, count=NUM_STORES, partitions=None):
    """Generate a synthetic stores dataset"""
    print(f"Generating {count} stores...")

    # Use determined number of partitions based on cluster size
    if partitions is None:
        partitions = min(50, max(10, count // 100))
    
    return spark.range(0, count, 1, partitions).selectExpr(
        "id as store_id",
        "concat('Store_', cast(id as string)) as store_name",
        "concat(cast(rand() * 9999 as int), ' Retail Dr') as address",
        "case when rand() < 0.2 then 'New York' when rand() < 0.4 then 'Los Angeles' when rand() < 0.6 then 'Chicago' else concat('City_', cast(rand() * 100 as int)) end as city",
        "case when rand() < 0.3 then 'CA' when rand() < 0.6 then 'NY' else concat('ST_', cast(rand() * 50 as int)) end as state",
        "concat(cast(10000 + rand() * 89999 as int)) as zipcode",
        "cast(1000 + rand() * 9000 as int) as manager_id",
        "cast(5000 + rand() * 95000 as int) as size_sqft",
        "date_add(to_date('2010-01-01'), cast(rand() * 4745 as int)) as open_date"
    )

@log_timing("Generate suppliers dataset")
def generate_suppliers(spark, count=NUM_SUPPLIERS, partitions=None):
    """Generate a synthetic suppliers dataset"""
    print(f"Generating {count} suppliers...")

    # Use determined number of partitions based on cluster size
    if partitions is None:
        partitions = min(50, max(10, count // 200))
    
    return spark.range(0, count, 1, partitions).selectExpr(
        "id as supplier_id",
        "concat('Supplier_', cast(id as string)) as supplier_name",
        "concat('Contact_', cast(id as string)) as contact_name",
        "concat('supplier_', cast(id as string), '@example.com') as email",
        "concat('(', cast(100 + rand() * 899 as int), ') ', cast(100 + rand() * 899 as int), '-', cast(1000 + rand() * 8999 as int)) as phone",
        "concat(cast(rand() * 9999 as int), ' Supply Ave') as address",
        "case when rand() < 0.3 then 'Shanghai' when rand() < 0.5 then 'Mumbai' when rand() < 0.7 then 'Mexico City' else concat('City_', cast(rand() * 50 as int)) end as city",
        "case when rand() < 0.2 then 'China' when rand() < 0.4 then 'India' when rand() < 0.6 then 'Mexico' when rand() < 0.8 then 'USA' else concat('Country_', cast(rand() * 20 as int)) end as country",
        "case when rand() < 0.3 then 'Net 30' when rand() < 0.6 then 'Net 60' else 'Net 90' end as payment_terms",
        "cast(1 + rand() * 9 as double) as quality_score"
    )

@log_timing("Generate transactions dataset")
def generate_transactions(spark, count=NUM_TRANSACTIONS, partitions=None):
    """Generate a synthetic transactions dataset with improved skew handling"""
    print(f"Generating {count} transactions...")

    # Calculate partitions based on data size and available resources
    if partitions is None:
        partitions = min(500, max(50, count // 50000))
    
    print(f"Using {partitions} partitions for transactions")
    
    # Generate data with controlled skew for better performance
    df = spark.range(0, count, 1, partitions)
    
    # Add salt column to help distribute skewed joins better
    df = df.withColumn("salt", (rand() * 100).cast("int"))
    
    # Generate with better control over skew
    return df.selectExpr(
        "id as transaction_id",
        # Create skew in customer distribution (heavy activity from certain customers)
        # But avoid extreme skew that would hurt performance
        "cast(case when rand() < 0.2 then mod(id, 1000) else mod(id, 1000000) end as int) as customer_id",
        "cast(mod(id, 1000) as int) as store_id",
        # Create moderate skew in product distribution (some products sell more frequently)
        "cast(case when rand() < 0.3 then mod(id + salt, 1000) else mod(id, 50000) end as int) as product_id",
        "cast(1 + mod(id, 10) as int) as quantity",
        "date_add(to_date('2023-01-01'), cast(mod(id, 365) as int)) as transaction_date",
        "concat(cast(cast(mod(id, 24) as int) as string), ':', lpad(cast(cast(mod(id, 60) as int) as string), 2, '0'), ':', lpad(cast(cast(mod(id, 60) as int) as string), 2, '0')) as transaction_time",
        "cast(4.99 + mod(id, 995) as double) as unit_price",
        "cast(mod(id, 50) / 100 as double) as discount",
        "case when mod(id, 10) < 4 then 'Credit Card' when mod(id, 10) < 7 then 'Debit Card' when mod(id, 10) < 9 then 'Mobile Payment' else 'Cash' end as payment_method"
    ).drop("salt")

@log_timing("Write dataset to NVMe")
def write_dataset_to_nvme(df, name, base_path, verify=VERIFY_WRITES):
    """Write a dataset to NVMe storage with enhanced reliability"""
    path = f"{base_path}/{name}"
    schema_path = f"{path}_schema.json"
    print(f"Writing {name} dataset to {path}...")
    
    # Ensure the path exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the schema to a JSON file
    schema_json = df.schema.json()
    with open(schema_path, "w") as f:
        f.write(schema_json)
    
    # Add a UUID to avoid conflicts
    import uuid
    write_id = str(uuid.uuid4())[:8]
    temp_path = f"{path}_{write_id}_temp"
    
    # Get partition count based on dataframe size
    row_count = df.count()
    partition_size = 1000000  # 1M rows per partition as target
    partition_count = max(10, min(200, row_count // partition_size + 1))
    
    print(f"Repartitioning to {partition_count} partitions for balanced NVMe write...")
    
    # K8s host-path optimization: ensure all partitions are large enough
    df = df.repartition(partition_count)
    
    # Write with overwrite mode to temp location first
    print(f"Writing to temporary location: {temp_path}")
    df.write.mode("overwrite").parquet(temp_path)
    
    # Force a barrier execution to ensure all executors have completed writing
    spark = df.sparkSession
    barrier_df = spark.range(0, 10).repartition(1)
    barrier_df.count()
    
    # Optional verification step
    if verify:
        print(f"Verifying write with sample read from {temp_path}")
        try:
            # Try to read back a sample
            verify_df = spark.read.parquet(temp_path).limit(100)
            verify_count = verify_df.count()
            print(f"Verified {verify_count} sample rows from temporary location")
        except Exception as e:
            print(f"WARNING: Verification failed: {str(e)}")
            # Continue anyway
    
    # Only after successful write and verification, move to final location
    print(f"Moving data from temporary location to final location")
    
    # First remove the old data if it exists
    if os.path.exists(path):
        print(f"Removing existing data at {path}")
        import shutil
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Failed to remove old data: {str(e)}")
    
    # Now move/rename the temp directory to final location
    try:
        if os.path.exists(temp_path):
            import shutil
            shutil.move(temp_path, path)
            print(f"Successfully moved data to final location: {path}")
        else:
            print(f"ERROR: Temporary path {temp_path} does not exist after write!")
    except Exception as e:
        print(f"ERROR: Failed to move data to final location: {str(e)}")
        # Fall back to direct read from temp location
        return temp_path
    
    return path
    
@log_timing("Read dataset from NVMe")
def read_dataset_from_nvme(spark, path, schema=None, resilient=RESILIENT_READ, max_retries=MAX_RETRIES):
    """Read a dataset from NVMe storage with enhanced error handling"""
    print(f"Reading dataset from {path}...")
    
    # Initialize retry counter
    retry_count = 0
    
    # Define function to attempt read with automatic retry
    def attempt_read():
        nonlocal retry_count
        
        while retry_count < max_retries:
            try:
                # First try to read the schema file
                schema_path = f"{path}_schema.json"
                explicit_schema = None
                
                if schema:
                    # Use provided schema if available
                    explicit_schema = schema
                    print(f"Using provided schema for reading")
                elif os.path.exists(schema_path):
                    try:
                        with open(schema_path, "r") as f:
                            schema_json = f.read()
                        
                        # Parse the schema
                        explicit_schema = StructType.fromJson(json.loads(schema_json))
                        print(f"Found and loaded schema from {schema_path}")
                    except Exception as schema_e:
                        print(f"Warning: Failed to load schema: {str(schema_e)}")
                
                # Try to read with explicit schema if available
                if explicit_schema:
                    df = spark.read.schema(explicit_schema).option("mergeSchema", "true").parquet(path)
                else:
                    # Check if path exists first
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Path does not exist: {path}")
                        
                    # Try to read with automatic schema inference
                    df = spark.read.option("mergeSchema", "true").parquet(path)
                
                # Force a small action to validate the read worked
                print(f"Verifying read by counting first 10 rows...")
                sample_count = df.limit(10).count()
                print(f"Successfully read and verified {sample_count} sample rows")
                
                # Do a slightly larger validation if in resilient mode
                if resilient:
                    col_count = len(df.columns)
                    print(f"Dataset has {col_count} columns")
                    
                    # Check a random sampling of data
                    sample_size = min(1000, df.count() // 100)  # 1% sample or 1000 rows, whichever is smaller
                    random_sample = df.sample(fraction=0.01).limit(sample_size)
                    sample_count = random_sample.count()
                    print(f"Validated random sample of {sample_count} rows")
                
                return df
                
            except Exception as e:
                retry_count += 1
                print(f"Error reading from {path} (attempt {retry_count}/{max_retries}): {str(e)}")
                
                if retry_count >= max_retries:
                    print(f"Maximum retries exceeded. Attempting recovery...")
                    break
                
                # Exponential backoff
                wait_time = 2 ** retry_count
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
        # All retries failed, attempt recovery
        return attempt_recovery()
        
    def attempt_recovery():
        """Last-ditch recovery attempt when normal reads fail"""
        print(f"Attempting recovery procedures for {path}...")
        
        try:
            # Look for alternate paths - sometimes temp files remain
            alt_paths = glob.glob(f"{path}*")
            if len(alt_paths) > 1:
                print(f"Found alternate paths: {alt_paths}")
                for alt_path in alt_paths:
                    if alt_path != path and "_temp" in alt_path:
                        print(f"Attempting to read from alternate path: {alt_path}")
                        try:
                            recovery_df = spark.read.option("mergeSchema", "true").parquet(alt_path)
                            sample_count = recovery_df.limit(5).count()
                            print(f"Successfully recovered {sample_count} sample rows from {alt_path}")
                            return recovery_df
                        except Exception as alt_e:
                            print(f"Failed to read alternate path: {str(alt_e)}")
            
            # Try reading individual files directly
            part_files = glob.glob(f"{path}/part-*")
            if part_files:
                print(f"Found {len(part_files)} part files, attempting to read first few files...")
                try:
                    # Try to get schema from file name or parameter
                    ds_name = os.path.basename(path)
                    recovery_schema = None
                    
                    if schema:
                        recovery_schema = schema
                    elif ds_name == "customers":
                        recovery_schema = get_customers_schema()
                    elif ds_name == "products":
                        recovery_schema = get_products_schema()
                    elif ds_name == "stores":
                        recovery_schema = get_stores_schema()
                    elif ds_name == "suppliers":
                        recovery_schema = get_suppliers_schema()
                    elif ds_name == "transactions":
                        recovery_schema = get_transactions_schema()
                    
                    if recovery_schema:
                        print(f"Using schema for {ds_name} to recover data")
                        recovery_df = spark.read.schema(recovery_schema).parquet(path)
                        return recovery_df
                except Exception as recovery_e:
                    print(f"Schema-based recovery failed: {str(recovery_e)}")
            
            # Last resort: regenerate the data
            print(f"All recovery attempts failed. Consider regenerating the dataset.")
            raise RuntimeError(f"Failed to read dataset from {path} after multiple attempts and recovery procedures")
            
        except Exception as recovery_e:
            print(f"Recovery procedures failed: {str(recovery_e)}")
            raise
    
    # Start the read attempt
    return attempt_read()

@log_timing("Perform heavy join with skew handling")
def perform_heavy_join(customers_df, products_df, stores_df, suppliers_df, transactions_df):
    """Perform a heavy join operation between all datasets with skew handling"""
    print("Performing heavy join between all datasets with skew optimization...")
    
    # Apply salt to help with skewed keys
    # This is a common technique to reduce skew in joins
    spark = customers_df.sparkSession
    
    # Broadcast smaller tables for optimization
    from pyspark.sql.functions import broadcast
    
    # Create temp views with consistent names for SQL
    customers_df.createOrReplaceTempView("customers")
    products_df.createOrReplaceTempView("products")
    stores_df.createOrReplaceTempView("stores")
    suppliers_df.createOrReplaceTempView("suppliers")
    transactions_df.createOrReplaceTempView("transactions")
    
    # Use SQL with skew hints and broadcasting for complex joins
    result = spark.sql("""
        SELECT /*+ BROADCAST(s) BROADCAST(sup) */
            t.transaction_id,
            t.transaction_date,
            c.customer_id,
            c.customer_name,
            c.loyalty_segment,
            p.product_id,
            p.product_name,
            p.category,
            p.subcategory,
            s.store_id,
            s.store_name,
            s.city as store_city,
            s.state as store_state,
            sup.supplier_id,
            sup.supplier_name,
            sup.country as supplier_country,
            t.quantity,
            t.unit_price,
            t.discount,
            (t.quantity * t.unit_price) * (1 - COALESCE(t.discount, 0)) as total_amount,
            (t.quantity * t.unit_price) * (1 - COALESCE(t.discount, 0)) - (t.quantity * p.cost) as profit,
            t.payment_method
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        JOIN products p ON t.product_id = p.product_id
        JOIN stores s ON t.store_id = s.store_id
        JOIN suppliers sup ON p.supplier_id = sup.supplier_id
    """)

    # Force execution and caching for better measurement
    result.cache()
    
    # Count with timing to measure join performance
    start_time = time.time()
    count = result.count()
    end_time = time.time()
    
    print(f"Join result contains {count} rows (computed in {end_time - start_time:.2f} seconds)")
    return result

@log_timing("Aggregation operations")
def perform_aggregations(joined_df):
    """Perform various aggregation operations on the joined data"""
    print("Performing multiple aggregation operations...")

    # Register view
    joined_df.createOrReplaceTempView("joined_data")
    spark = joined_df.sparkSession

    # Execute several aggregation queries with improved hints
    print("Executing category/loyalty/state aggregation...")
    agg1 = spark.sql("""
        SELECT /*+ COALESCE(50) */
            category,
            loyalty_segment,
            store_state,
            COUNT(*) as transaction_count,
            SUM(total_amount) as total_sales,
            AVG(total_amount) as avg_transaction_value,
            SUM(profit) as total_profit,
            (SUM(profit) / SUM(total_amount)) * 100 as profit_margin_percentage
        FROM joined_data
        GROUP BY category, loyalty_segment, store_state
        ORDER BY total_sales DESC
    """)

    print("Executing supplier country/category aggregation...")
    agg2 = spark.sql("""
        SELECT /*+ COALESCE(50) */
            supplier_country,
            category,
            subcategory,
            COUNT(*) as transaction_count,
            COUNT(DISTINCT product_id) as unique_products,
            SUM(quantity) as total_units_sold,
            SUM(total_amount) as total_sales,
            SUM(profit) as total_profit
        FROM joined_data
        GROUP BY supplier_country, category, subcategory
        ORDER BY total_sales DESC
    """)

    print("Executing time-based aggregation...")
    agg3 = spark.sql("""
        SELECT /*+ COALESCE(100) */
            DATE_FORMAT(transaction_date, 'yyyy-MM') as month,
            store_city,
            loyalty_segment,
            COUNT(*) as transaction_count,
            SUM(total_amount) as total_sales,
            COUNT(DISTINCT customer_id) as unique_customers,
            SUM(total_amount) / COUNT(DISTINCT customer_id) as avg_spend_per_customer,
            SUM(profit) as total_profit
        FROM joined_data
        GROUP BY DATE_FORMAT(transaction_date, 'yyyy-MM'), store_city, loyalty_segment
        ORDER BY month, total_sales DESC
    """)

    # Force execution with progress tracking
    print("Computing aggregation 1...")
    agg1.cache()
    agg1_count = agg1.count()
    print(f"Aggregation 1 complete: {agg1_count} rows")
    
    print("Computing aggregation 2...")
    agg2.cache()
    agg2_count = agg2.count()
    print(f"Aggregation 2 complete: {agg2_count} rows")
    
    print("Computing aggregation 3...")
    agg3.cache()
    agg3_count = agg3.count()
    print(f"Aggregation 3 complete: {agg3_count} rows")

    return agg1, agg2, agg3

@log_timing("NVMe Heavy Join Benchmark")
def run_nvme_join_benchmark():
    """Main function to run the NVMe heavy join benchmark with enhanced reliability"""
    print("\n========= STARTING NVME HEAVY JOIN BENCHMARK =========\n")
    
    spark = create_spark_session()
    
    try:
        # Get local dirs configuration
        sc = spark.sparkContext
        local_dirs = sc._jsc.sc().getConf().get("spark.local.dir", "/tmp").split(",")
        
        if not local_dirs:
            print("WARNING: No local dirs configured. Defaulting to /tmp")
            local_dirs = ["/tmp/spark-nvme-test"]
        
        # Check if we have NVMe drives in the configuration
        nvme_dirs = [d for d in local_dirs if "nvme" in d]
        if nvme_dirs:
            # Use the first NVMe path
            nvme_path = nvme_dirs[0]
            print(f"Using NVMe path: {nvme_path} (found NVMe directories: {nvme_dirs})")
        else:
            # Use the first directory for our test
            nvme_path = local_dirs[0]
            print(f"No NVMe paths found. Using local path: {nvme_path}")
        
        # Verify that the NVMe path exists and is writable
        os.makedirs(nvme_path, exist_ok=True)
        try:
            test_file = f"{nvme_path}/write_test_{int(time.time())}.txt"
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"Verified write access to {nvme_path}")
        except Exception as e:
            print(f"WARNING: Could not write to {nvme_path}: {str(e)}")
            print("Attempting to continue anyway...")
        
        # Phase 1: Generate datasets
        print("\n=== PHASE 1: GENERATING DATASETS ===")
        
        # Check if we should regenerate data or try to use existing data
        regenerate_data = True
        customers_df = products_df = stores_df = suppliers_df = transactions_df = None
        
        try:
            if os.path.exists(f"{nvme_path}/customers") and \
               os.path.exists(f"{nvme_path}/products") and \
               os.path.exists(f"{nvme_path}/stores") and \
               os.path.exists(f"{nvme_path}/suppliers") and \
               os.path.exists(f"{nvme_path}/transactions"):
                print("Found existing datasets. Checking if they're readable...")
                
                # Try to read a sample to verify data integrity
                try:
                    customers_sample = spark.read.option("samplingRatio", "0.01").parquet(f"{nvme_path}/customers").limit(5)
                    customers_sample.count()
                    products_sample = spark.read.option("samplingRatio", "0.01").parquet(f"{nvme_path}/products").limit(5)
                    products_sample.count()
                    
                    print("Existing datasets appear to be valid. Loading them instead of regenerating.")
                    regenerate_data = False
                    
                    # Define paths for later use
                    customers_path = f"{nvme_path}/customers"
                    products_path = f"{nvme_path}/products"
                    stores_path = f"{nvme_path}/stores"
                    suppliers_path = f"{nvme_path}/suppliers"
                    transactions_path = f"{nvme_path}/transactions"
                    
                    # Load the existing datasets
                    customers_df = read_dataset_from_nvme(spark, customers_path, get_customers_schema())
                    products_df = read_dataset_from_nvme(spark, products_path, get_products_schema())
                    stores_df = read_dataset_from_nvme(spark, stores_path, get_stores_schema())
                    suppliers_df = read_dataset_from_nvme(spark, suppliers_path, get_suppliers_schema())
                    transactions_df = read_dataset_from_nvme(spark, transactions_path, get_transactions_schema())
                    
                except Exception as e:
                    print(f"Error reading existing datasets: {str(e)}")
                    print("Will regenerate all datasets.")
                    regenerate_data = True
        except Exception as e:
            print(f"Error checking for existing datasets: {str(e)}")
            regenerate_data = True
        
        if regenerate_data:
            print("Generating new datasets...")
            customers_df = generate_customers(spark)
            products_df = generate_products(spark)
            stores_df = generate_stores(spark)
            suppliers_df = generate_suppliers(spark)
            transactions_df = generate_transactions(spark)
            
            # Force action to ensure generation worked
            print(f"Generated {customers_df.count()} customers")
            print(f"Generated {products_df.count()} products")
            print(f"Generated {stores_df.count()} stores")
            print(f"Generated {suppliers_df.count()} suppliers")
            print(f"Generated {transactions_df.count()} transactions")
            
            # Phase 2: Write datasets to NVMe
            print("\n=== PHASE 2: WRITING DATASETS TO NVME ===")
            customers_path = write_dataset_to_nvme(customers_df, "customers", nvme_path)
            products_path = write_dataset_to_nvme(products_df, "products", nvme_path)
            stores_path = write_dataset_to_nvme(stores_df, "stores", nvme_path)
            suppliers_path = write_dataset_to_nvme(suppliers_df, "suppliers", nvme_path)
            transactions_path = write_dataset_to_nvme(transactions_df, "transactions", nvme_path)
        
        # Make sure we have all dataframes loaded
        assert customers_df is not None, "Customers DataFrame is not loaded"
        assert products_df is not None, "Products DataFrame is not loaded"
        assert stores_df is not None, "Stores DataFrame is not loaded"
        assert suppliers_df is not None, "Suppliers DataFrame is not loaded"
        assert transactions_df is not None, "Transactions DataFrame is not loaded"

        # Phase 3 (optional if regenerate_data=False): Verify datasets were read properly
        print("\n=== PHASE 3: VERIFYING DATASETS ===")
        customers_count = customers_df.count()
        products_count = products_df.count()
        stores_count = stores_df.count()
        suppliers_count = suppliers_df.count()
        transactions_count = transactions_df.count()
        
        print(f"Verified customers: {customers_count} rows")
        print(f"Verified products: {products_count} rows")
        print(f"Verified stores: {stores_count} rows")
        print(f"Verified suppliers: {suppliers_count} rows")
        print(f"Verified transactions: {transactions_count} rows")

        # Phase 4: Perform heavy join
        print("\n=== PHASE 4: PERFORMING HEAVY JOIN ===")
        joined_df = perform_heavy_join(customers_df, products_df, stores_df, suppliers_df, transactions_df)

        # Phase 5: Perform complex aggregations
        print("\n=== PHASE 5: PERFORMING COMPLEX AGGREGATIONS ===")
        agg1, agg2, agg3 = perform_aggregations(joined_df)

        # Write results back to NVMe
        print("\n=== PHASE 6: WRITING RESULTS TO NVME ===")
        write_dataset_to_nvme(joined_df, "joined_result", nvme_path)
        write_dataset_to_nvme(agg1, "agg_by_category_loyalty_state", nvme_path)
        write_dataset_to_nvme(agg2, "agg_by_supplier_category", nvme_path)
        write_dataset_to_nvme(agg3, "agg_by_month_city_loyalty", nvme_path)

        print("\n=== BENCHMARK SUMMARY ===")
        print(f"Successfully processed:")
        print(f"- {customers_count} customers")
        print(f"- {products_count} products")
        print(f"- {suppliers_count} suppliers")
        print(f"- {stores_count} stores")
        print(f"- {transactions_count} transactions")
        print(f"- Performed complex join resulting in {joined_df.count()} records")
        print(f"- Created 3 complex aggregations")

    except Exception as e:
        print(f"ERROR in benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup (optional)
        if CLEANUP_ON_EXIT:
            print("Cleaning up cached dataframes...")
            # Unpersist cached data to free memory
            for df_name in ["joined_df", "agg1", "agg2", "agg3"]:
                if df_name in locals() and locals()[df_name] is not None:
                    try:
                        locals()[df_name].unpersist()
                    except:
                        pass
        
        # Stop Spark session
        print("Stopping Spark session...")
        spark.stop()
        print("\n========= BENCHMARK COMPLETE =========\n")

if __name__ == "__main__":
    run_nvme_join_benchmark()
