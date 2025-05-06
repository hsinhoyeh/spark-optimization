#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skewed Dataset with Array Explosion Example for Apache Spark
-----------------------------------------------------------
This script demonstrates how to generate and handle a skewed dataset 
with hundreds of fields and array-type data for explosion operations.
"""

import time
import os
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, 
    DoubleType, ArrayType, MapType, BooleanType, TimestampType
)
from pyspark.sql.functions import (
    col, explode, lit, expr, rand, array, 
    when, broadcast, count, sum, avg, 
    spark_partition_id, hash
)

# Configuration
NUM_RECORDS = 1_000_000  # Base number of records
NUM_FIELDS = 300        # Number of fields per record
ARRAY_MIN_SIZE = 100     # Minimum array size
ARRAY_MAX_SIZE = 1000000    # Maximum array size
SKEW_PERCENTAGE = 0.4   # Percentage of records with skewed data
RESHUFFLE_PARTITIONS = 48  # Number of partitions to use for reshuffling
ENABLE_RESHUFFLE = True  # Toggle for reshuffling before explode

# Define a decorator for timing
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
    """Create and configure a Spark session optimized for large skewed datasets"""
    return (SparkSession.builder
            .appName("Skewed Dataset with Array Explosion")
            # Enable adaptive query execution
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
            .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
            .config("spark.sql.autoBroadcastJoinThreshold", "10MB")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.maxResultSize", "4g") \
            
            # Shuffle partitions based on data size
            .config("spark.sql.shuffle.partitions", "40")
            
            # Memory and execution configurations
            .config("spark.executor.memory", "16g")
            .config("spark.driver.memory", "12g")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "4g")
            
            # Optimized I/O
            .config("spark.io.compression.codec", "lz4")
            .config("spark.reducer.maxSizeInFlight", "96m")
            
            # Fault tolerance
            .config("spark.task.maxFailures", "5")
            
            .getOrCreate())


@log_timing("Generate skewed array field")
def generate_skewed_array_field(df):
    """
    Generate an array field with genuinely skewed distribution
    """
    print("Generating array field with properly skewed distribution...")
    
    # Create a column with truly variable array sizes - using a proper power law distribution
    # This formula will create a strong skew where most records have small arrays, but some have very large ones
    df_with_sizes = df.withColumn("array_sizes", 
        when(rand() < SKEW_PERCENTAGE,
             # Skewed records: use power law distribution to create extreme outliers
             # Dividing by a small random number creates a long-tailed distribution
             (lit(ARRAY_MIN_SIZE * 10) / (rand() * lit(0.1) + lit(0.01))).cast("int")
        ).otherwise(
             # Non-skewed records: use uniform distribution for smaller arrays
             (rand() * lit(ARRAY_MIN_SIZE) + lit(ARRAY_MIN_SIZE / 2)).cast("int")
        ))
    
    # Cap the maximum size to prevent extreme outliers causing OOM
    df_with_sizes = df_with_sizes.withColumn("array_sizes",
        when(col("array_sizes") > lit(ARRAY_MAX_SIZE), lit(ARRAY_MAX_SIZE)).otherwise(col("array_sizes"))
    )
    
    # Verify the distribution by showing some statistics
    print("\nAnalyzing array size distribution:")
    df_with_sizes.select("array_sizes").summary().show()
    
    # Compute and show percentiles explicitly
    percentiles = df_with_sizes.selectExpr(
        "approx_percentile(array_sizes, 0.5) as p50",
        "approx_percentile(array_sizes, 0.9) as p90",
        "approx_percentile(array_sizes, 0.99) as p99",
        "max(array_sizes) as max"
    ).collect()[0]
    
    p50 = percentiles["p50"] or 1  # Avoid division by zero
    p90 = percentiles["p90"] or 1
    p99 = percentiles["p99"] or 1
    p_max = percentiles["max"] or 1
    
    print(f"\nVerifying skew in array sizes:")
    print(f"50th percentile (median): {p50}")
    print(f"90th percentile: {p90}")
    print(f"99th percentile: {p99}")
    print(f"Maximum size: {p_max}")
    
    print(f"\nSkew verification - percentile ratios:")
    print(f"P90/P50 ratio: {p90/p50:.2f} (values > 2 suggest skew)")
    print(f"P99/P50 ratio: {p99/p50:.2f} (values > 5 suggest significant skew)")
    print(f"Max/P50 ratio: {p_max/p50:.2f} (high values indicate extreme outliers)")
    
    if p99/p50 < 5:
        print("WARNING: Generated data is not sufficiently skewed - adjust parameters!")
    else:
        print("VERIFIED: Generated data shows proper skew distribution")
    
    # Show frequency distribution
    print("\nShow frequency distribution of array sizes:")
    df_with_sizes.groupBy("array_sizes").count().orderBy("array_sizes").show(20)
    
    # Function to create a simple histogram of the distribution
    print("\nGenerating histogram of array sizes...")
    size_counts = df_with_sizes.groupBy("array_sizes").count().collect()
    size_dict = {row["array_sizes"]: row["count"] for row in size_counts}
    size_range = list(range(min(size_dict.keys()), min(100, max(size_dict.keys()))))
    
    print("Array Size Histogram (first 100 sizes):")
    for size in sorted(size_dict.keys())[:20]:  # Show just first 20 for brevity
        count = size_dict[size]
        bar = "#" * min(50, int(count / max(size_dict.values()) * 50))
        print(f"{size:4d}: {bar} ({count})")
    
    # Start with an empty array
    result_df = df_with_sizes.withColumn("items_array", array())
    
    # Process arrays in small batches
    batch_size = 50  # Process in small batches
    max_elements = 1000  # Maximum number of elements to generate
    
    # Track the element index
    current_idx = 0
    
    print(f"Building arrays incrementally up to {max_elements} elements in batches of {batch_size}...")
    
    # Process batch by batch
    while current_idx < max_elements:
        end_idx = min(current_idx + batch_size, max_elements)
        
        print(f"Processing elements {current_idx} to {end_idx-1}...")
        
        # Create a batch of elements as a new array
        batch_elements = []
        
        for i in range(current_idx, end_idx):
            # This creates an array with elements based on the randomly generated array_sizes
            element_expr = f"""
            case when {i} < array_sizes then
                named_struct(
                    'element_id', id * 1000 + {i},
                    'element_name', concat('item_', cast(id as string), '_', cast({i} as string)),
                    'element_value', rand() * 100,
                    'element_category', case 
                                          when rand() < 0.2 then 'Category A'
                                          when rand() < 0.5 then 'Category B'
                                          when rand() < 0.8 then 'Category C'
                                          else 'Category D'
                                       end,
                    'element_tags', array(
                                     concat('tag_', cast((id % 10) as string))
                                   ),
                    'element_active', (id + {i}) % 2 = 0,
                    'element_score', rand() * 1000,
                    'element_date', date_add(to_date('2023-01-01'), cast(rand() * 365 as int))
                )
            else null end
            """
            batch_elements.append(element_expr)
        
        # Create the batch array expression
        batch_expr = f"array({', '.join(batch_elements)})"
        
        # Add this batch of elements
        result_df = result_df.withColumn("batch_array", expr(batch_expr))
        
        # Filter out null values from the batch
        result_df = result_df.withColumn("batch_array", 
                         expr("filter(batch_array, x -> x is not null)"))
        
        # Concatenate with existing array
        result_df = result_df.withColumn("items_array", 
                         expr("concat(items_array, batch_array)"))
        
        # Drop temporary column
        result_df = result_df.drop("batch_array")
        
        # Checkpoint occasionally to avoid building too big of a lineage
        if current_idx > 0 and current_idx % 200 == 0:
            print(f"Checkpointing at element {current_idx}...")
            result_df = result_df.localCheckpoint()
        
        # Move to next batch
        current_idx = end_idx
    
    # Verify final array size distribution
    final_size_df = result_df.select(
        expr("size(items_array)").alias("final_size")
    )
    
    print("\nFinal array size distribution after all processing:")
    final_size_df.summary().show()
    
    # Calculate final percentile ratios
    final_percentiles = final_size_df.selectExpr(
        "approx_percentile(final_size, 0.5) as p50",
        "approx_percentile(final_size, 0.9) as p90",
        "approx_percentile(final_size, 0.99) as p99"
    ).collect()[0]
    
    final_p50 = final_percentiles["p50"] or 1  # Avoid division by zero
    final_p90 = final_percentiles["p90"] or 1
    final_p99 = final_percentiles["p99"] or 1
    
    print(f"\nFinal skew verification - percentile ratios:")
    print(f"P90/P50 ratio: {final_p90/final_p50:.2f} (values > 2 suggest skew)")
    print(f"P99/P50 ratio: {final_p99/final_p50:.2f} (values > 5 suggest significant skew)")
    
    if final_p99/final_p50 < 5:
        print("WARNING: Final data is not sufficiently skewed!")
    else:
        print("SUCCESS: Final data shows proper skew distribution")
    
    # Cleanup and return
    result_df = result_df.drop("array_sizes")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return result_df

    # Define a UDF to create arrays of varying sizes with the skew we want
def generate_base_dataset(spark, num_records=NUM_RECORDS):
    """Generate a base dataset with hundreds of fields and skewed data distribution"""
    print(f"Generating base dataset with {num_records} records and {NUM_FIELDS} fields...")
    
    # Calculate partitions based on record count
    partitions = min(500, max(20, num_records // 100000))
    print(f"Using {partitions} partitions for data generation")
    
    # Start with a basic dataframe
    df = spark.range(0, num_records, 1, partitions)
    
    # Add salt for better distribution
    df = df.withColumn("salt", (rand() * 100).cast("int"))
    
    # Add ID field
    df = df.withColumn("record_id", col("id"))
    
    # Add timestamp field
    df = df.withColumn("creation_time", 
        expr("date_add(to_timestamp('2023-01-01 00:00:00'), cast(rand() * 365 as int))"))
    
    # Generate 195+ additional fields to reach NUM_FIELDS
    # Different field types to simulate real-world data
    
    # Add string fields (about 50)
    for i in range(1, 51):
        df = df.withColumn(f"string_field_{i}", 
            expr(f"concat('value_', cast(((id + {i} * salt) % 10000) as string))"))
    
    # Add integer fields (about 50)
    for i in range(1, 51):
        df = df.withColumn(f"int_field_{i}", 
            expr(f"cast(((id * {i} + salt) % 1000000) as int)"))
    
    # Add double fields (about 50)
    for i in range(1, 51):
        df = df.withColumn(f"double_field_{i}", 
            expr(f"cast(((id * {i} * 0.01) + (salt * 0.1)) as double)"))
    
    # Add boolean fields (about 30)
    for i in range(1, 31):
        df = df.withColumn(f"bool_field_{i}", 
            expr(f"(id + {i} + salt) % {(i+1)*2} = 0"))
    
    # Add key fields with skewed distribution (about 20)
    for i in range(1, 21):
        # Create skew pattern where certain values appear much more frequently
        skew_factor = i * 10
        df = df.withColumn(f"key_field_{i}", 
            when(rand() < SKEW_PERCENTAGE, 
                 expr(f"id % {skew_factor}")
            ).otherwise(
                expr(f"id % {num_records / skew_factor}")
            ).cast("int"))
    
    # Generate the array field using the FIXED skewed generator
    df = generate_skewed_array_field(df)
    
    # Drop intermediate columns
    df = df.drop("salt", "id")
    
    return df

@log_timing("Generate array field")
def generate_array_field(df):
    """
    Generate the array field with skewed distribution using a step-by-step approach
    that avoids creating massive expression strings
    """
    print("Generating array field with memory-optimized approach...")
    
    # First, create a column with the desired array sizes
    # But cap the maximum size to prevent OOM
    max_safe_size = 1000  # Significantly reduce max array size for testing
    
    df_with_sizes = df.withColumn("array_sizes", 
        when(rand() < SKEW_PERCENTAGE,
             # Skewed records have large arrays but cap at max_safe_size
             (rand() * (max_safe_size - ARRAY_MIN_SIZE) + ARRAY_MIN_SIZE).cast("int")
        ).otherwise(
             # Non-skewed records have smaller arrays
             (rand() * 10 + ARRAY_MIN_SIZE).cast("int")
        ))
    
    # Instead of one massive expression, we'll build small arrays incrementally
    
    # Step 1: Create a base array with just a few elements
    base_size = 10  # Start with just 10 elements
    base_array_expr = """
    array(
    """
    
    for i in range(base_size):
        if i > 0:
            base_array_expr += ","
        base_array_expr += f"""
        case when {i} < array_sizes then
            named_struct(
                'element_id', id * 1000 + {i},
                'element_name', concat('item_', cast(id as string), '_', cast({i} as string)),
                'element_value', rand() * 100,
                'element_category', case 
                                      when rand() < 0.2 then 'Category A'
                                      when rand() < 0.5 then 'Category B'
                                      when rand() < 0.8 then 'Category C'
                                      else 'Category D'
                                   end,
                'element_tags', array(
                                 concat('tag_', cast((id % 10) as string))
                               ),
                'element_active', (id + {i}) % 2 = 0,
                'element_score', rand() * 1000,
                'element_date', date_add(to_date('2023-01-01'), cast(rand() * 365 as int))
            )
        else
            null
        end"""
    
    base_array_expr += """
    )
    """
    
    # Apply initial expression for base array
    df_with_base = df_with_sizes.withColumn("items_array", 
                        expr(base_array_expr))
    
    # Filter out null values
    df_with_base = df_with_base.withColumn("items_array", 
                        expr("filter(items_array, x -> x is not null)"))
    
    # Function to create a small batch of elements
    def create_batch_expr(start_idx, batch_size):
        """Create expression for a small batch of array elements"""
        batch_expr = "array("
        for i in range(batch_size):
            idx = start_idx + i
            if i > 0:
                batch_expr += ","
            batch_expr += f"""
            case when {idx} < array_sizes then
                named_struct(
                    'element_id', id * 1000 + {idx},
                    'element_name', concat('item_', cast(id as string), '_', cast({idx} as string)),
                    'element_value', rand() * 100,
                    'element_category', case 
                                          when rand() < 0.2 then 'Category A'
                                          when rand() < 0.5 then 'Category B'
                                          when rand() < 0.8 then 'Category C'
                                          else 'Category D'
                                       end,
                    'element_tags', array(
                                     concat('tag_', cast((id % 10) as string))
                                   ),
                    'element_active', (id + {idx}) % 2 = 0,
                    'element_score', rand() * 1000,
                    'element_date', date_add(to_date('2023-01-01'), cast(rand() * 365 as int))
                )
            else
                null
            end"""
        batch_expr += ")"
        return batch_expr
    
    # Incrementally add more elements in small batches
    # This avoids creating huge strings that cause OOM
    result_df = df_with_base
    
    # Determine how many batches to process
    batch_size = 50  # Process in small batches
    start_idx = base_size
    max_idx = min(max_safe_size, ARRAY_MAX_SIZE)
    
    print(f"Building arrays incrementally up to {max_idx} elements in batches of {batch_size}...")
    
    # Process batch by batch
    while start_idx < max_idx:
        end_idx = min(start_idx + batch_size, max_idx)
        current_batch_size = end_idx - start_idx
        
        print(f"Processing elements {start_idx} to {end_idx-1}...")
        
        # Create this batch of elements
        batch_expr = create_batch_expr(start_idx, current_batch_size)
        
        # Add to existing array
        result_df = result_df.withColumn("batch_array", expr(batch_expr))
        
        # Filter null values from batch
        result_df = result_df.withColumn("batch_array", 
                         expr("filter(batch_array, x -> x is not null)"))
        
        # Concatenate with existing array
        result_df = result_df.withColumn("items_array", 
                         expr("concat(items_array, batch_array)"))
        
        # Drop temporary column
        result_df = result_df.drop("batch_array")
        
        # Checkpoint occasionally to avoid building too big of a lineage
        if (start_idx - base_size) % 200 == 0 and start_idx > base_size:
            print(f"Checkpointing at element {start_idx}...")
            result_df = result_df.localCheckpoint()
        
        # Move to next batch
        start_idx = end_idx
    
    # Cleanup and return
    result_df = result_df.drop("array_sizes")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return result_df

@log_timing("Write dataset to storage")
def write_dataset(df, output_path, format="parquet"):
    """Write the dataset to storage with optimized settings"""
    print(f"Writing dataset to {output_path}...")
    
    # Ensure the path exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Repartition for balanced write
    row_count = df.count()
    partition_size = 100000  # Target rows per partition
    partition_count = max(10, min(200, row_count // partition_size + 1))
    df = df.repartition(partition_count)
    
    # Write with optimized settings
    df.write.format(format)\
            .mode("overwrite")\
            .option("compression", "snappy")\
            .save(output_path)
    
    print(f"Dataset saved to {output_path}")
    return output_path

@log_timing("Read dataset from storage")
def read_dataset(spark, path, format="parquet"):
    """Read the dataset from storage"""
    print(f"Reading dataset from {path}...")
    return spark.read.format(format).load(path)

@log_timing("Bucketed Array Explosion")
def bucketed_array_explosion(df, num_buckets=10):
    """
    Implement a bucketing strategy for handling skewed arrays during explosion.
    
    This approach:
    1. Assigns records to buckets based on array size
    2. Processes each bucket with appropriate parallelism
    3. Combines results from all buckets
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with the array column to explode
    num_buckets : int
        Number of buckets to create based on array size
        
    Returns:
    --------
    DataFrame
        Exploded DataFrame with balanced partitions
    dict
        Performance metrics
    """
    # Import necessary functions
    import math
    from pyspark.sql.functions import udf, col, expr
    from pyspark.sql.types import IntegerType
    
    print(f"Processing array data using bucketing approach with {num_buckets} buckets...")
    
    # Start timing
    bucketing_start = time.time()
    
    # First, analyze the array sizes to determine distribution
    print("Analyzing array size distribution...")
    size_df = df.select(
        "record_id",
        expr("size(items_array)").alias("array_size")
    )
    
    # Get statistics about array sizes
    size_stats = size_df.agg(
        expr("min(array_size)").alias("min_size"),
        expr("max(array_size)").alias("max_size"),
        expr("avg(array_size)").alias("avg_size"),
        expr("approx_percentile(array_size, array(0.25, 0.5, 0.75, 0.9, 0.95, 0.99))").alias("percentiles")
    ).collect()[0]
    
    min_size = size_stats["min_size"]
    max_size = size_stats["max_size"]
    avg_size = size_stats["avg_size"]
    p25, p50, p75, p90, p95, p99 = size_stats["percentiles"]
    
    print(f"Array size statistics:")
    print(f"- Min: {min_size}")
    print(f"- 25th percentile: {p25}")
    print(f"- Median: {p50}")
    print(f"- 75th percentile: {p75}")
    print(f"- 90th percentile: {p90}")
    print(f"- 95th percentile: {p95}")
    print(f"- 99th percentile: {p99}")
    print(f"- Max: {max_size}")
    print(f"- Mean: {avg_size:.2f}")
    
    # 1. Create bucket boundaries using exponential scale
    # This gives more fine-grained control for smaller arrays
    # and broader buckets for larger arrays
    
    # Define bucket boundaries using a mixture of linear and exponential scales
    if max_size <= p99 * 2:
        # If no extreme outliers, use linear scale
        bucket_step = (max_size - min_size) / num_buckets
        boundaries = [min_size + i * bucket_step for i in range(num_buckets + 1)]
    else:
        # With extreme outliers, use a combination approach
        # First 80% of buckets linearly distributed up to p95
        # Last 20% of buckets exponentially distributed from p95 to max
        linear_buckets = int(num_buckets * 0.8)
        exp_buckets = num_buckets - linear_buckets
        
        # Linear part (up to p95)
        linear_step = (p95 - min_size) / linear_buckets
        linear_boundaries = [min_size + i * linear_step for i in range(linear_buckets)]
        
        # Exponential part (p95 to max)
        # Using logarithmic scale to create exponential bucket boundaries
        if exp_buckets > 0:
            log_min = math.log(p95 + 1)  # +1 to handle zero
            log_max = math.log(max_size + 1)
            log_step = (log_max - log_min) / exp_buckets
            exp_boundaries = [math.exp(log_min + i * log_step) - 1 for i in range(1, exp_buckets + 1)]
            boundaries = linear_boundaries + exp_boundaries
        else:
            boundaries = linear_boundaries + [max_size]
    
    # Ensure max_size is included
    if boundaries[-1] < max_size:
        boundaries[-1] = max_size
    
    # Round boundaries for readability
    boundaries = [int(b) for b in boundaries]
    
    print(f"Created {len(boundaries)-1} bucket boundaries: {boundaries}")
    
    # 2. Assign each record to a bucket
    print("Assigning records to buckets based on array size...")
    
    # Create a bucketing UDF
    def assign_bucket(array_size):
        for i, boundary in enumerate(boundaries[1:], 1):
            if array_size <= boundary:
                return i - 1
        return len(boundaries) - 2  # Last bucket
    
    # Register UDF
    bucket_udf = udf(assign_bucket, IntegerType())
    
    # Apply bucketing
    bucketed_df = df.withColumn("array_size", expr("size(items_array)"))
    bucketed_df = bucketed_df.withColumn("bucket_id", bucket_udf(col("array_size")))
    
    # 3. Create temporary views for SQL operations
    bucketed_df.createOrReplaceTempView("bucketed_arrays")
    
    # 4. Process each bucket with appropriate parallelism
    print("Processing buckets with tailored parallelism...")
    
    # Calculate bucket statistics
    bucket_stats = bucketed_df.groupBy("bucket_id").agg(
        expr("count(*)").alias("count"),
        expr("min(array_size)").alias("min_size"),
        expr("max(array_size)").alias("max_size"),
        expr("avg(array_size)").alias("avg_size")
    ).orderBy("bucket_id").collect()
    
    print("\nBucket statistics:")
    print("Bucket ID | Count | Min Size | Max Size | Avg Size")
    print("-" * 60)
    for stat in bucket_stats:
        bucket_id = stat["bucket_id"]
        count = stat["count"]
        min_size = stat["min_size"]
        max_size = stat["max_size"]
        avg_size = stat["avg_size"]
        print(f"{bucket_id:9d} | {count:5d} | {min_size:8d} | {max_size:8d} | {avg_size:8.2f}")
    
    # Initialize empty result DataFrame
    all_exploded = None
    
    # Process each bucket
    for bucket_id in range(len(boundaries) - 1):
        # Extract records for this bucket
        bucket_query = f"SELECT * FROM bucketed_arrays WHERE bucket_id = {bucket_id}"
        bucket_df = df.sparkSession.sql(bucket_query)
        
        # Skip empty buckets
        if bucket_df.isEmpty():
            continue
        
        bucket_count = bucket_df.count()
        print(f"Processing bucket {bucket_id} with {bucket_count} records...")
        
        # Calculate appropriate parallelism for this bucket
        # More records with larger arrays need more partitions
        avg_size = next((s["avg_size"] for s in bucket_stats if s["bucket_id"] == bucket_id), 0)
        
        # Scale partitions based on average array size and record count
        size_factor = avg_size / p50 if p50 > 0 else 1
        record_factor = bucket_count / (df.count() / num_buckets) if df.count() > 0 else 1
        
        # Calculate number of partitions with a minimum of 1
        bucket_partitions = max(1, int(num_buckets * size_factor * record_factor))
        
        # For very large arrays, increase parallelism further
        if avg_size > p95:
            bucket_partitions *= 2
        
        print(f"Using {bucket_partitions} partitions for bucket {bucket_id}")
        
        # Repartition based on a hash of record_id to spread load
        if bucket_partitions > 1:
            bucket_df = bucket_df.repartition(bucket_partitions, "record_id")
        
        # Explode the array
        bucket_exploded = bucket_df.selectExpr(
            "record_id", 
            "inline(items_array)"
        )
        
        # Append to result
        if all_exploded is None:
            all_exploded = bucket_exploded
        else:
            all_exploded = all_exploded.union(bucket_exploded)
    
    # Handle the case where no buckets were processed
    if all_exploded is None:
        print("Warning: No buckets were processed. Creating an empty result DataFrame.")
        all_exploded = df.sparkSession.createDataFrame(
            [],
            df.selectExpr("record_id", "inline(items_array)").schema
        )
    
    # Cache the result
    all_exploded.cache()
    
    # Count to materialize
    bucket_count = all_exploded.count()
    
    # Calculate duration
    bucketing_end = time.time()
    bucketing_duration = bucketing_end - bucketing_start
    
    print(f"Bucketed explosion completed in {bucketing_duration:.2f} seconds, yielding {bucket_count} records")
    
    # Analyze partition distribution
    try:
        print("\nAnalyzing partition distribution:")
        partition_counts = all_exploded.groupBy(spark_partition_id()).count()
        
        partition_stats = partition_counts.agg(
            expr("min(count)").alias("min"),
            expr("avg(count)").alias("mean"),
            expr("max(count)").alias("max"),
            expr("stddev(count)").alias("stddev")
        ).collect()[0]
        
        min_count = partition_stats["min"]
        mean_count = partition_stats["mean"]
        max_count = partition_stats["max"]
        stddev_count = partition_stats["stddev"]
        
        # Calculate coefficient of variation
        cv = stddev_count / mean_count if mean_count > 0 else 0
        
        print(f"Partition statistics:")
        print(f"- Min: {min_count}")
        print(f"- Mean: {mean_count:.2f}")
        print(f"- Max: {max_count}")
        print(f"- Stddev: {stddev_count:.2f}")
        print(f"- CV: {cv:.4f} (lower is better - indicates more balanced partitions)")
        
        # Calculate imbalance ratio
        imbalance = max_count / mean_count if mean_count > 0 else 0
        print(f"- Max/Mean ratio: {imbalance:.2f}x (lower is better)")
        
    except Exception as e:
        print(f"Could not analyze partition distribution: {str(e)}")
        cv = float('inf')
    
    # Return results
    metrics = {
        "duration": bucketing_duration,
        "count": bucket_count,
        "cv": cv
    }
    
    return all_exploded, metrics

@log_timing("Reshuffle data by array size")
def reshuffle_by_array_size(df):
    """
    Reshuffle the data to balance workload across workers based on array size
    This helps prevent workers from getting overloaded during the explode operation
    """
    print("Reshuffling data to balance workload before explosion...")
    
    # Add column with array size
    df_with_size = df.withColumn("array_size", expr("size(items_array)"))
    
    # Calculate statistics for balancing
    size_stats = df_with_size.select(
        avg("array_size").alias("avg_size"),
        expr("percentile(array_size, 0.95)").alias("p95_size"),
        expr("percentile(array_size, 0.99)").alias("p99_size"),
        expr("max(array_size)").alias("max_size")
    ).collect()[0]
    
    avg_size = size_stats["avg_size"]
    max_size = size_stats["max_size"]
    p95_size = size_stats["p95_size"]
    
    print(f"Array size statistics - Avg: {avg_size:.2f}, 95th percentile: {p95_size}, Max: {max_size}")
    
    # Define a weight factor for each record based on its array size
    # Records with larger arrays get higher weights and will be distributed more evenly
    df_with_weight = df_with_size.withColumn(
        "distribution_weight", 
        when(col("array_size") > p95_size, 
             # Records with very large arrays get more weight
             expr("ceil(array_size / greatest(1.0, array_size / 10))")
        ).otherwise(
             # Normal records get weight proportional to array size
             expr("ceil(array_size / greatest(1.0, array_size / 2))")
        )
    )
    
    # Calculate a distribution key that will be used for partitioning
    # This ensures that records with larger arrays are distributed across more partitions
    df_with_key = df_with_weight.withColumn(
        "distribution_key",
        # Hash combination of record_id and a modulo based on weight
        # This helps spread out records with large arrays
        expr("hash(concat(cast(record_id as string), cast(pmod(record_id, distribution_weight) as string)))")
    )
    
    # Repartition based on distribution key to spread the workload
    reshuffled_df = df_with_key.repartition(RESHUFFLE_PARTITIONS, "distribution_key")
    
    # Drop the temporary columns
    reshuffled_df = reshuffled_df.drop("array_size", "distribution_weight", "distribution_key")
    
    return reshuffled_df

@log_timing("Explode array and analyze skew")
def explode_and_analyze(df, enable_reshuffle=ENABLE_RESHUFFLE):
    """Explode the array field and analyze the resulting skew"""
    print(f"Processing array data with reshuffle {'enabled' if enable_reshuffle else 'disabled'}...")
    
    # Reduce the complexity of the dataset before operations
    # Extract only the array size information first
    print("Extracting array size information...")
    size_df = df.select(
        "record_id",
        expr("size(items_array)").alias("array_size")
    )
    
    # Analyze the array sizes before explosion
    size_stats = size_df.describe().collect()
    print("\nArray size statistics before explosion:")
    for row in size_stats:
        print(f"{row['summary']}: {row['array_size']}")
    
    # Calculate standard deviation and coefficient of variation for skew analysis
    from pyspark.sql.functions import mean, stddev
    std_stats = size_df.agg(
        mean("array_size").alias("mean_len"),
        stddev("array_size").alias("stddev_len")
    ).collect()[0]
    
    mean_len = std_stats["mean_len"]
    stddev_len = std_stats["stddev_len"]
    
    # Calculate coefficient of variation (CV) - normalized measure of dispersion
    cv = stddev_len / mean_len if mean_len > 0 else 0
    
    print(f"\nStandard deviation metrics:")
    print(f"Mean array length: {mean_len:.2f}")
    print(f"Standard deviation: {stddev_len:.2f}")
    print(f"Coefficient of variation: {cv:.4f} (higher values indicate greater relative variability)")
    
    # Calculate percentile ratios to quantify skew
    percentiles = size_df.selectExpr(
        "percentile_approx(array_size, 0.5) as p50",
        "percentile_approx(array_size, 0.9) as p90",
        "percentile_approx(array_size, 0.99) as p99",
        "max(array_size) as max"
    ).collect()[0]
    
    p50 = percentiles["p50"]
    p90 = percentiles["p90"]
    p99 = percentiles["p99"]
    p_max = percentiles["max"]
    
    print(f"\nPercentile ratios:")
    print(f"P99/P50 ratio: {p99/p50 if p50 > 0 else 'N/A':.2f} (values > 2 suggest significant skew)")
    print(f"Max/P50 ratio: {p_max/p50 if p50 > 0 else 'N/A':.2f} (high values indicate extreme outliers)")
    print(f"P90/P50 ratio: {p90/p50 if p50 > 0 else 'N/A':.2f} (values > 2 suggest skew)")
    
    # Identify records with large arrays (potential skew causes)
    large_arrays = size_df.filter(col("array_size") > 50).count()
    total_records = size_df.count()
    print(f"\nRecords with large arrays (>50 elements): {large_arrays} out of {total_records} ({100.0 * large_arrays / total_records:.2f}%)")
    
    # Add detailed skewness measurement - distribution of array sizes
    print("\nDetailed distribution of array sizes (top 20 most frequent):")
    size_df.groupBy("array_size").count().orderBy(col("count").desc()).show(20, truncate=False)
    
    # Generate histogram of array sizes to visualize distribution
    print("\nGenerating histogram of array sizes to visualize skewness...")
    try:
        # Create histogram buckets with exponential boundaries to better show skew
        buckets = [0]
        current = 10
        while current < ARRAY_MAX_SIZE:
            buckets.append(current)
            current = current * 2  # Double each time
        buckets.append(ARRAY_MAX_SIZE + 1)
        
        histogram_df = size_df.select(
            expr(f"width_bucket(array_size, array({','.join(map(str, buckets))}))").alias("bucket"),
            "array_size"
        )
        
        # Get bucket statistics
        histogram_stats = histogram_df.groupBy("bucket").agg(
            count("*").alias("count"),
            min("array_size").alias("min_size"),
            max("array_size").alias("max_size")
        ).orderBy("bucket")
        
        # Convert to a more readable format
        print("Array size histogram (exponential buckets):")
        print("Bucket Range             | Count      | % of Total")
        print("-" * 60)
        
        histogram_rows = histogram_stats.collect()
        for row in histogram_rows:
            bucket_num = row["bucket"]
            if bucket_num == 0:
                # Handle items outside the range
                range_str = f"< {buckets[1]}"
            elif bucket_num >= len(buckets) - 1:
                range_str = f">= {buckets[-2]}"
            else:
                range_str = f"{buckets[bucket_num-1]} to {buckets[bucket_num]-1}"
            
            count = row["count"]
            percentage = 100.0 * count / total_records
            
            # Format the output
            print(f"{range_str.ljust(23)} | {str(count).ljust(10)} | {percentage:.2f}%")
    except Exception as e:
        print(f"Could not generate histogram: {str(e)}")
    
    # Calculate skewness metrics
    try:
        from pyspark.sql.functions import skewness, kurtosis
        skew_metrics = size_df.select(
            skewness("array_size").alias("skewness"),
            kurtosis("array_size").alias("kurtosis")
        ).collect()[0]
        print(f"\nSkewness metrics:")
        print(f"Skewness: {skew_metrics['skewness']:.4f} (>0 means right-skewed, <0 means left-skewed)")
        print(f"Kurtosis: {skew_metrics['kurtosis']:.4f} (>0 means heavy-tailed distribution)")
        
        # Calculate Gini coefficient as another measure of skewness/inequality
        print("\nCalculating Gini coefficient for array size distribution...")
        
        # Collect all array sizes for Gini calculation
        # For large datasets, we might want to sample this
        sample_fraction = min(1.0, 10000.0 / total_records)  # Sample at most 10,000 records
        size_sample = size_df.sample(fraction=sample_fraction).collect()
        
        # Extract array sizes from the sample
        array_sizes = [row["array_size"] for row in size_sample]
        
        # Calculate Gini coefficient
        def calculate_gini(values):
            """Calculate the Gini coefficient of a set of values"""
            if len(values) <= 1 or sum(values) == 0:
                return 0
            
            # Sort values
            sorted_values = sorted(values)
            # Calculate cumulative sum
            cum_values = [0]
            for v in sorted_values:
                cum_values.append(cum_values[-1] + v)
            
            # Calculate Gini coefficient
            n = len(sorted_values)
            B = sum(cum_values[1:]) / (cum_values[-1] * n)
            return 1 - 2 * B
        
        gini = calculate_gini(array_sizes)
        print(f"Gini coefficient: {gini:.4f} (0 = perfect equality, 1 = complete inequality)")
        
    except Exception as e:
        print(f"Could not calculate skewness metrics: {str(e)}")
    
    # Simplest possible explode for both approaches to avoid compiler errors
    if enable_reshuffle:
        print("\n=== PERFORMANCE COMPARISON: WITH VS WITHOUT RESHUFFLE ===\n")
        
        # First approach - without reshuffling
        try:
            print("Running baseline explosion (WITHOUT reshuffle)...")
            baseline_start = time.time()
            
            # Extremely simplified explosion to avoid compiler errors
            baseline_exploded = df.selectExpr(
                "record_id",
                "inline(items_array)"  # Using inline instead of explode for simplicity
            )
            
            baseline_count = baseline_exploded.count()
            baseline_end = time.time()
            baseline_duration = baseline_end - baseline_start
            
            print(f"Baseline explosion completed in {baseline_duration:.2f} seconds, yielding {baseline_count} records")
            
            # Create a simple version with just a few fields for analysis
            baseline_simple = baseline_exploded.select(
                "record_id", 
                "element_id",
                "element_category",
                "element_score"
            )
            
            # Save the metrics
            baseline_metrics = {
                "duration": baseline_duration,
                "count": baseline_count
            }
            
            # Clean up
            baseline_exploded.unpersist()
        except Exception as e:
            print(f"Error during baseline explosion: {str(e)}")
            print("Unable to complete baseline measurement, continuing with optimized approach only")
            baseline_metrics = {
                "duration": float('inf'),
                "count": 0
            }
            baseline_simple = None
            
        # Second approach - with reshuffling
        try:
            print("\nRunning optimized explosion (WITH reshuffle)...")
            optimized_start = time.time()
            
            # Do the reshuffling
            print("Redistributing data based on array sizes...")
            
            # Simplified reshuffle: just add a distribution key based on array size
            # This helps spread out records with large arrays across partitions
            reshuffled_df = df.withColumn(
                "array_size", expr("size(items_array)")
            ).withColumn(
                "distribution_key", expr("mod(hash(concat(cast(record_id as string), cast(array_size as string))), 100)")
            ).repartition(50, "distribution_key")
            
            # Now explode with the same simplified approach for fair comparison
            optimized_exploded = reshuffled_df.selectExpr(
                "record_id",
                "inline(items_array)"
            )
            
            optimized_count = optimized_exploded.count()
            optimized_end = time.time()
            optimized_duration = optimized_end - optimized_start
            
            print(f"Optimized explosion completed in {optimized_duration:.2f} seconds, yielding {optimized_count} records")
            
            # Create simplified version for analysis
            optimized_simple = optimized_exploded.select(
                "record_id", 
                "element_id",
                "element_category",
                "element_score"
            )
            
            # Save metrics
            optimized_metrics = {
                "duration": optimized_duration,
                "count": optimized_count
            }
            
            # Clean up
            reshuffled_df.unpersist()
            optimized_exploded.unpersist()
        except Exception as e:
            print(f"Error during optimized explosion: {str(e)}")
            print("Unable to complete optimized measurement")
            optimized_metrics = {
                "duration": float('inf'),
                "count": 0
            }
            optimized_simple = None
            
        # Choose which version to use for further analysis
        if baseline_simple is not None and optimized_simple is not None:
            # Both succeeded, report comparisons
            speedup = baseline_metrics["duration"] / max(0.001, optimized_metrics["duration"])
            improvement_pct = (baseline_metrics["duration"] - optimized_metrics["duration"]) / baseline_metrics["duration"] * 100
            
            print(f"\nPerformance comparison results:")
            print(f"- Baseline: {baseline_metrics['duration']:.2f} seconds for {baseline_metrics['count']} records")
            print(f"- Optimized: {optimized_metrics['duration']:.2f} seconds for {optimized_metrics['count']} records")
            if speedup > 1:
                print(f"- Result: Reshuffling was faster by {speedup:.2f}x ({improvement_pct:.2f}% improvement)")
            else:
                print(f"- Result: Reshuffling was slower by {1/speedup:.2f}x ({-improvement_pct:.2f}% slower)")
            
            # Use the faster approach for continuing
            if speedup > 1:
                print("Using the optimized (reshuffled) dataset for further analysis")
                exploded_df = optimized_simple
                comparison_results = {
                    "baseline": baseline_metrics,
                    "optimized": optimized_metrics,
                    "winner": "optimized",
                    "speedup": speedup
                }
            else:
                print("Using the baseline dataset for further analysis")
                exploded_df = baseline_simple
                comparison_results = {
                    "baseline": baseline_metrics,
                    "optimized": optimized_metrics,
                    "winner": "baseline",
                    "speedup": speedup
                }
        elif baseline_simple is not None:
            # Only baseline succeeded
            print("Using baseline dataset for further analysis (optimized approach failed)")
            exploded_df = baseline_simple
            comparison_results = {
                "baseline": baseline_metrics,
                "optimized": optimized_metrics,
                "winner": "baseline",
                "speedup": float('inf')
            }
        elif optimized_simple is not None:
            # Only optimized succeeded
            print("Using optimized dataset for further analysis (baseline approach failed)")
            exploded_df = optimized_simple
            comparison_results = {
                "baseline": baseline_metrics,
                "optimized": optimized_metrics,
                "winner": "optimized",
                "speedup": 0
            }
        else:
            # Neither succeeded, create a super-simplified version
            print("WARNING: Both approaches failed. Creating a minimal dataset for analysis.")
            exploded_df = df.select(
                "record_id", 
                explode(col("items_array")).alias("item")
            ).select(
                "record_id",
                col("item.element_id").alias("element_id")
            )
            comparison_results = {
                "baseline": {"duration": float('inf'), "count": 0},
                "optimized": {"duration": float('inf'), "count": 0},
                "winner": "none",
                "speedup": 0
            }
    else:
        # No comparison needed, just do the explosion with a simplified approach
        print("Performing explosion without reshuffling...")
        
        try:
            # Use the inline approach which tends to be more stable
            exploded_df = df.selectExpr(
                "record_id",
                "inline(items_array)"
            ).select(
                "record_id",
                "element_id",
                "element_category",
                "element_score"
            )
            comparison_results = None
        except Exception as e:
            print(f"Error during explosion: {str(e)}")
            print("Trying an ultra-simplified approach...")
            
            # Last resort approach
            exploded_df = df.select(
                "record_id",
                explode(col("items_array")).alias("item")
            ).select(
                "record_id",
                col("item.element_id").alias("element_id")
            )
            comparison_results = None
    
    # Cache result for further analysis
    exploded_df.cache()
    
    # Start analysis
    print("\n=== ANALYZING EXPLOSION RESULTS ===")
    
    # Count records
    total_rows = exploded_df.count()
    print(f"Total rows after explosion: {total_rows}")
    
    # Try to get partition distribution
    try:
        print("\nAnalyzing partition distribution:")
        partition_counts = exploded_df.groupBy(spark_partition_id()).count()
        partition_summary = partition_counts.summary("min", "25%", "50%", "75%", "max").collect()
        
        print("Partition size statistics:")
        for row in partition_summary:
            print(f"{row['summary']}: {row['count']}")
            
        # Get top 5 largest partitions
        print("\nLargest partitions:")
        partition_counts.orderBy(col("count").desc()).show(5)
    except Exception as e:
        print(f"Could not analyze partition distribution: {str(e)}")
    
    # Analyze record distribution
    try:
        print("\nAnalyzing records per source record:")
        record_counts = exploded_df.groupBy("record_id").count()
        record_summary = record_counts.summary("min", "25%", "50%", "75%", "max").collect()
        
        print("Elements per record statistics:")
        for row in record_summary:
            print(f"{row['summary']}: {row['count']}")
            
        # Get top records with most elements
        print("\nRecords with most elements:")
        record_counts.orderBy(col("count").desc()).show(10)
        
        # Calculate the skew factor if possible
        try:
            max_elements = float(record_summary[4]["count"])  # 'max' is at index 4
            avg_elements = record_counts.select(expr("avg(count)")).collect()[0][0]
            skew_factor = max_elements / avg_elements
            print(f"Skew factor (max/avg): {skew_factor:.2f}x")
        except Exception as skew_e:
            print(f"Could not calculate skew factor: {str(skew_e)}")
    except Exception as e:
        print(f"Could not analyze records distribution: {str(e)}")
    
    # Return both the data and comparison results if available
    if comparison_results:
        return exploded_df, comparison_results
    else:
        return exploded_df

@log_timing("FlatMap transformation with skew handling")
def flatmap_with_skew_handling(df):
    """
    Use flatMap transformation with built-in skew handling to process array data efficiently
    """
    print("Processing array data using flatMap with skew handling...")
    
    # First, examine the data distribution
    print("Analyzing array size distribution...")
    size_df = df.select(
        "record_id",
        expr("size(items_array)").alias("array_size")
    )
    
    # Get array size statistics
    size_stats = size_df.agg(
        expr("avg(array_size)").alias("mean_size"),
        expr("stddev(array_size)").alias("stddev_size"),
        expr("percentile(array_size, 0.99)").alias("p99_size"),
        expr("max(array_size)").alias("max_size")
    ).collect()[0]
    
    mean_size = size_stats["mean_size"]
    p99_size = size_stats["p99_size"]
    max_size = size_stats["max_size"]
    
    print(f"Array size statistics - Mean: {mean_size:.2f}, 99th percentile: {p99_size}, Max: {max_size}")
    
    # Define a skew threshold - arrays larger than this will get special handling
    skew_threshold = p99_size  # Use 99th percentile as the threshold
    print(f"Using skew threshold of {skew_threshold} elements")
    
    # Start timing the flatMap approach
    print("\nRunning flatMap with skew handling...")
    flatmap_start = time.time()
    
    # 1. First, separate normal and skewed records
    normal_records = df.filter(expr(f"size(items_array) <= {skew_threshold}"))
    skewed_records = df.filter(expr(f"size(items_array) > {skew_threshold}"))
    
    # Count both datasets
    normal_count = normal_records.count()
    skewed_count = skewed_records.count()
    
    print(f"Normal records: {normal_count} ({normal_count*100.0/(normal_count+skewed_count):.2f}%)")
    print(f"Skewed records: {skewed_count} ({skewed_count*100.0/(normal_count+skewed_count):.2f}%)")
    
    # 2. Process normal records using standard flatMap
    # Convert to RDD for flatMap operation
    normal_rdd = normal_records.select("record_id", "items_array").rdd
    
    # Define the flatMap function for normal records
    def flatten_normal_array(row):
        record_id = row[0]
        items = row[1]
        results = []
        
        # Skip processing if items is None
        if not items:
            return results
            
        # Process each element in the array
        for item in items:
            # Extract fields from the struct
            element_id = item.element_id if hasattr(item, 'element_id') else None
            element_name = item.element_name if hasattr(item, 'element_name') else None
            element_value = item.element_value if hasattr(item, 'element_value') else None
            element_category = item.element_category if hasattr(item, 'element_category') else None
            element_active = item.element_active if hasattr(item, 'element_active') else None
            element_score = item.element_score if hasattr(item, 'element_score') else None
            
            # Create a new row
            results.append((
                record_id,
                element_id,
                element_name,
                element_value,
                element_category,
                element_active,
                element_score
            ))
        
        return results
    
    # Apply the flatMap transformation to normal records
    normal_flattened_rdd = normal_rdd.flatMap(flatten_normal_array)
    
    # 3. Process skewed records with special handling
    # For highly skewed records, we'll use a different approach that
    # processes each record's array in smaller chunks and repartitions
    
    # Convert skewed records to RDD
    skewed_rdd = skewed_records.select("record_id", "items_array").rdd
    
    # Define the flatMap function with chunking for skewed records
    def flatten_skewed_array(row):
        record_id = row[0]
        items = row[1]
        results = []
        
        # Skip processing if items is None
        if not items:
            return results
        
        # Add a chunk ID to help distribute the work
        chunk_size = 100  # Process in chunks of 100 elements
        
        # Process each element in the array
        for i, item in enumerate(items):
            chunk_id = i // chunk_size
            
            # Extract fields from the struct
            element_id = item.element_id if hasattr(item, 'element_id') else None
            element_name = item.element_name if hasattr(item, 'element_name') else None
            element_value = item.element_value if hasattr(item, 'element_value') else None
            element_category = item.element_category if hasattr(item, 'element_category') else None
            element_active = item.element_active if hasattr(item, 'element_active') else None
            element_score = item.element_score if hasattr(item, 'element_score') else None
            
            # Create a new row with chunk ID
            results.append((
                record_id,
                element_id,
                element_name,
                element_value,
                element_category,
                element_active,
                element_score,
                chunk_id  # Add chunk ID for better distribution
            ))
        
        return results
    
    # Apply the flatMap transformation to skewed records
    skewed_flattened_rdd = skewed_rdd.flatMap(flatten_skewed_array)
    
    # Repartition the skewed data by chunk ID to distribute more evenly
    if not skewed_flattened_rdd.isEmpty():
        skewed_flattened_rdd = skewed_flattened_rdd.map(
            lambda x: (x[7], x)  # Use chunk_id as the key
        ).partitionBy(200).map(lambda x: x[1][:-1])  # Remove chunk_id from final result
    
    # 4. Union the normal and skewed results if both exist
    if normal_flattened_rdd.isEmpty():
        combined_rdd = skewed_flattened_rdd
    elif skewed_flattened_rdd.isEmpty():
        combined_rdd = normal_flattened_rdd
    else:
        combined_rdd = normal_flattened_rdd.union(skewed_flattened_rdd)
    
    # Define the schema for the flattened data
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
    
    schema = StructType([
        StructField("record_id", IntegerType(), True),
        StructField("element_id", IntegerType(), True),
        StructField("element_name", StringType(), True),
        StructField("element_value", DoubleType(), True),
        StructField("element_category", StringType(), True),
        StructField("element_active", BooleanType(), True),
        StructField("element_score", DoubleType(), True)
    ])
    
    # Create DataFrame from RDD with the defined schema
    flattened_df = df.sparkSession.createDataFrame(combined_rdd, schema)
    
    # Cache the result for further analysis
    flattened_df.cache()
    
    # Count the rows to materialize the cache
    flatmap_count = flattened_df.count()
    
    # Calculate duration
    flatmap_end = time.time()
    flatmap_duration = flatmap_end - flatmap_start
    
    print(f"FlatMap with skew handling completed in {flatmap_duration:.2f} seconds, yielding {flatmap_count} records")
    
    # Analyze partition distribution
    try:
        print("\nAnalyzing partition distribution:")
        partition_counts = flattened_df.groupBy(spark_partition_id()).count()
        partition_summary = partition_counts.agg(
            expr("min(count)").alias("min"),
            expr("percentile(count, 0.25)").alias("p25"),
            expr("percentile(count, 0.5)").alias("p50"),
            expr("percentile(count, 0.75)").alias("p75"),
            expr("max(count)").alias("max")
        ).collect()[0]
        
        print("Partition size statistics:")
        print(f"Min: {partition_summary['min']}")
        print(f"25th percentile: {partition_summary['p25']}")
        print(f"Median: {partition_summary['p50']}")
        print(f"75th percentile: {partition_summary['p75']}")
        print(f"Max: {partition_summary['max']}")
        
        # Calculate coefficient of variation for partition sizes
        partition_stats = partition_counts.agg(
            expr("avg(count)").alias("mean_size"),
            expr("stddev(count)").alias("stddev_size")
        ).collect()[0]
        
        mean_size = partition_stats["mean_size"]
        stddev_size = partition_stats["stddev_size"]
        partition_cv = stddev_size / mean_size if mean_size > 0 else 0
        
        print(f"Partition balance (CV): {partition_cv:.4f} (lower is better - indicates more balanced partitions)")
    except Exception as e:
        print(f"Could not analyze partition distribution: {str(e)}")
    
    # Return the results
    flatmap_metrics = {
        "duration": flatmap_duration,
        "count": flatmap_count
    }
    
    return flattened_df, flatmap_metrics

@log_timing("Salted Array Explosion")
def salted_array_explosion(df, num_salts=10):
    """
    Implement an optimized salting strategy for handling skewed arrays during explosion.
    
    This approach:
    1. Uses adaptive salting based on array size
    2. Applies different salt counts for different-sized arrays
    3. Optimizes partition balance with weighted distribution
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with the array column to explode
    num_salts : int
        Base number of salt partitions to use (will be scaled for larger arrays)
        
    Returns:
    --------
    DataFrame
        Exploded DataFrame with balanced partitions
    dict
        Performance metrics
    """
    print(f"Processing array data using optimized salting approach...")
    
    # Set min and max salts based on the provided num_salts
    min_salts = num_salts
    max_salts = max(num_salts * 10, 100)  # Scale up to handle extreme skew
    
    # Start timing
    salting_start = time.time()
    
    # First, analyze the array sizes to determine skew
    print("Analyzing array size distribution...")
    size_df = df.select(
        "record_id",
        expr("size(items_array)").alias("array_size")
    )
    
    # Get statistics about array sizes
    size_stats = size_df.agg(
        expr("min(array_size)").alias("min_size"),
        expr("avg(array_size)").alias("mean_size"),
        expr("percentile(array_size, 0.5)").alias("p50_size"),
        expr("percentile(array_size, 0.9)").alias("p90_size"),
        expr("percentile(array_size, 0.95)").alias("p95_size"),
        expr("percentile(array_size, 0.99)").alias("p99_size"),
        expr("max(array_size)").alias("max_size")
    ).collect()[0]
    
    min_size = size_stats["min_size"]
    mean_size = size_stats["mean_size"]
    p50_size = size_stats["p50_size"]
    p90_size = size_stats["p90_size"]
    p95_size = size_stats["p95_size"]
    p99_size = size_stats["p99_size"]
    max_size = size_stats["max_size"]
    
    print(f"Array size statistics:")
    print(f"- Min: {min_size}")
    print(f"- Mean: {mean_size:.2f}")
    print(f"- Median (p50): {p50_size}")
    print(f"- p90: {p90_size}")
    print(f"- p95: {p95_size}")
    print(f"- p99: {p99_size}")
    print(f"- Max: {max_size}")
    
    # Calculate skew ratios (larger values indicate more skew)
    skew_ratio_p99_p50 = p99_size / p50_size if p50_size > 0 else 0
    skew_ratio_max_p50 = max_size / p50_size if p50_size > 0 else 0
    
    print(f"Skew ratios:")
    print(f"- p99/p50: {skew_ratio_p99_p50:.2f}x")
    print(f"- max/p50: {skew_ratio_max_p50:.2f}x")
    
    # Define multiple thresholds with adaptive salt counts
    # We'll define 4 categories of arrays:
    # 1. Small arrays - no salting needed
    # 2. Medium arrays - minimal salting
    # 3. Large arrays - moderate salting
    # 4. Extremely large arrays - aggressive salting
    
    # Define thresholds based on percentiles
    small_threshold = p50_size  # Use median as small threshold
    medium_threshold = p90_size  # p90 for medium
    large_threshold = p99_size   # p99 for large
    
    # Define salt counts for each category
    # Adaptive salting based on skew severity
    base_salt_count = min_salts
    medium_salt_count = min(max_salts, max(min_salts, int(base_salt_count * min(3, skew_ratio_p99_p50 / 3))))
    large_salt_count = min(max_salts, max(min_salts, int(base_salt_count * min(5, skew_ratio_p99_p50))))
    extreme_salt_count = min(max_salts, max(min_salts, int(base_salt_count * min(10, skew_ratio_max_p50))))
    
    print(f"Thresholds and salting strategy:")
    print(f"- Small arrays (<= {small_threshold}): No salting")
    print(f"- Medium arrays (<= {medium_threshold}): {medium_salt_count} salts")
    print(f"- Large arrays (<= {large_threshold}): {large_salt_count} salts")
    print(f"- Extremely large arrays (> {large_threshold}): {extreme_salt_count} salts")
    
    # Create a UDF to determine salt count based on array size
    def determine_salt_count(array_size):
        if array_size <= small_threshold:
            return 0  # No salting for small arrays
        elif array_size <= medium_threshold:
            return medium_salt_count
        elif array_size <= large_threshold:
            return large_salt_count
        else:
            return extreme_salt_count
    
    # Register as UDF
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType
    salt_count_udf = udf(determine_salt_count, IntegerType())
    
    # Add column with array size and salt count
    df_with_size = df.withColumn("array_size", expr("size(items_array)"))
    df_with_salt_count = df_with_size.withColumn("salt_count", salt_count_udf(df_with_size["array_size"]))
    
    # 1. Split dataset into categories
    small_df = df_with_salt_count.filter("salt_count = 0")
    salted_df = df_with_salt_count.filter("salt_count > 0")
    
    # Count records in each category
    small_count = small_df.count()
    salted_count = salted_df.count()
    
    total_count = small_count + salted_count
    small_pct = (small_count / total_count) * 100 if total_count > 0 else 0
    salted_pct = (salted_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Record distribution:")
    print(f"- Small arrays (no salting): {small_count} ({small_pct:.2f}%)")
    print(f"- Arrays requiring salting: {salted_count} ({salted_pct:.2f}%)")
    
    # 2. For small arrays, just use standard explode
    print("Processing small arrays with standard explode...")
    if small_count > 0:
        small_exploded = small_df.selectExpr(
            "record_id",
            "inline(items_array)"
        )
    else:
        # Create empty DataFrame with same schema if no small records
        small_exploded = df.sparkSession.createDataFrame(
            [],
            df.selectExpr("record_id", "inline(items_array)").schema
        )
    
    # 3. For arrays that need salting, use an improved approach with adaptive salting
    if salted_count > 0:
        print(f"Applying adaptive salting strategy...")
        
        # Create salted copies with varying salt counts
        # First, explode the salt counts to create the right number of salts per record
        salted_df.createOrReplaceTempView("salted_arrays")
        
        # Use SQL for more control and efficiency
        try:
            # Try more advanced SQL approach first for better performance
            exploded_salts_df = df.sparkSession.sql("""
            WITH exploded_salts AS (
                SELECT 
                    record_id,
                    items_array,
                    salt_count,
                    -- Generate sequence of salt values from 0 to salt_count-1
                    posexplode(sequence(0, salt_count - 1)) as (pos, salt_value)
                FROM salted_arrays
            )
            SELECT 
                record_id,
                items_array,
                salt_value,
                -- Generate deterministic but well-distributed partition key based on 
                -- both record_id and salt_value for better balance
                hash(concat(cast(record_id as string), '-', cast(salt_value as string))) % 1000 as partition_key
            FROM exploded_salts
            """)
        except Exception as e:
            print(f"Advanced SQL approach failed: {e}")
            print("Falling back to simpler approach...")
            
            # Fallback approach if the SQL with sequence generation fails
            from pyspark.sql.functions import explode, lit, array
            
            # Create a temporary column with an array of salt values
            # For each record, create an array of size salt_count filled with the same record_id
            salted_df = salted_df.withColumn(
                "salt_values", 
                expr("transform(sequence(0, salt_count - 1), x -> x)")
            )
            
            # Explode the salt values
            exploded_salts_df = salted_df.select(
                "record_id", 
                "items_array",
                explode("salt_values").alias("salt_value")
            ).withColumn(
                "partition_key",
                expr("hash(concat(cast(record_id as string), '-', cast(salt_value as string))) % 1000")
            )
        
        # Get distribution of partition keys to check balance
        partition_key_counts = exploded_salts_df.groupBy("partition_key").count()
        partition_key_stats = partition_key_counts.agg(
            expr("min(count)").alias("min"),
            expr("avg(count)").alias("avg"),
            expr("max(count)").alias("max"),
            expr("stddev(count)").alias("stddev")
        ).collect()[0]
        
        print(f"Salt distribution before explosion:")
        print(f"- Min: {partition_key_stats['min']}")
        print(f"- Avg: {partition_key_stats['avg']:.2f}")
        print(f"- Max: {partition_key_stats['max']}")
        print(f"- Stddev: {partition_key_stats['stddev']:.2f}")
        print(f"- CV: {partition_key_stats['stddev']/partition_key_stats['avg'] if partition_key_stats['avg'] > 0 else float('inf'):.4f} (lower is better)")
        
        # Repartition based on partition_key for better distribution
        # Scale partitions based on dataset size and skew
        num_partitions = min(2000, max(100, int(num_salts * skew_ratio_max_p50)))
        print(f"Using {num_partitions} partitions for salted processing")
        
        exploded_salts_df = exploded_salts_df.repartition(num_partitions, "partition_key")
        
        # Now perform the explosion with better parallelism
        salted_exploded = exploded_salts_df.selectExpr(
            "record_id", 
            "inline(items_array)"
        ).drop("salt_value", "partition_key")
        
        # Remove duplicates that might be created by the salting process
        # Only needed for element_id column which should be unique per record
        salted_exploded = salted_exploded.dropDuplicates(["record_id", "element_id"])
    else:
        # Create empty DataFrame with same schema if no salted records
        salted_exploded = df.sparkSession.createDataFrame(
            [],
            df.selectExpr("record_id", "inline(items_array)").schema
        )
    
    # 4. Union the results
    all_exploded = small_exploded.union(salted_exploded)
    
    # Cache the result for analysis
    all_exploded.cache()
    
    # Trigger computation and count
    salted_count = all_exploded.count()
    
    # Calculate duration
    salting_end = time.time()
    salting_duration = salting_end - salting_start
    
    print(f"Optimized salted explosion completed in {salting_duration:.2f} seconds, yielding {salted_count} records")
    
    # Analyze partition distribution
    try:
        print("\nAnalyzing final partition distribution:")
        partition_counts = all_exploded.groupBy(spark_partition_id()).count()
        
        partition_stats = partition_counts.agg(
            expr("min(count)").alias("min"),
            expr("avg(count)").alias("mean"),
            expr("max(count)").alias("max"),
            expr("stddev(count)").alias("stddev")
        ).collect()[0]
        
        min_count = partition_stats["min"]
        mean_count = partition_stats["mean"]
        max_count = partition_stats["max"]
        stddev_count = partition_stats["stddev"]
        
        # Calculate coefficient of variation
        cv = stddev_count / mean_count if mean_count > 0 else 0
        
        print(f"Partition statistics:")
        print(f"- Min: {min_count}")
        print(f"- Mean: {mean_count:.2f}")
        print(f"- Max: {max_count}")
        print(f"- Stddev: {stddev_count:.2f}")
        print(f"- CV: {cv:.4f} (lower is better - indicates more balanced partitions)")
        
        # Calculate imbalance ratio
        imbalance = max_count / mean_count if mean_count > 0 else 0
        print(f"- Max/Mean ratio: {imbalance:.2f}x (lower is better)")
        
    except Exception as e:
        print(f"Could not analyze partition distribution: {str(e)}")
        cv = float('inf')
    
    # Return results
    metrics = {
        "duration": salting_duration,
        "count": salted_count,
        "cv": cv
    }
    
    return all_exploded, metrics

@log_timing("Union-Based Array Explosion")
def union_based_explosion(df):
    """
    Implement a union-based strategy for handling extremely skewed arrays during explosion.
    
    This approach:
    1. Splits very large arrays into multiple chunks
    2. Explodes each chunk separately
    3. Unions the results
    
    This is particularly effective for datasets with extreme skew where a few records
    have arrays that are orders of magnitude larger than the average.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with the array column to explode
        
    Returns:
    --------
    DataFrame
        Exploded DataFrame with balanced partitions
    dict
        Performance metrics
    """
    # Import necessary functions
    from pyspark.sql.functions import col, expr, spark_partition_id
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
    
    # First, analyze the array sizes to determine skew
    print("Analyzing array size distribution...")
    size_df = df.select(
        "record_id",
        expr("size(items_array)").alias("array_size")
    )
    
    # Get statistics about array sizes using percentiles
    try:
        percentiles = df.selectExpr(
            "approx_percentile(size(items_array), array(0.9, 0.95, 0.99)) as percentiles"
        ).collect()[0]
        
        p90, p95, p99 = percentiles["percentiles"]
        
        # Set chunk size dynamically based on percentiles
        # Use a safe default if percentiles calculation fails
        max_array_chunk = max(100, min(int(p95), 1000))  # safe, balanced heuristic
    except Exception as e:
        print(f"Warning: Error calculating percentiles: {str(e)}")
        print("Using default chunk size of 500")
        # Fallback to a reasonable default
        max_array_chunk = 500
        p90, p95, p99 = 0, 0, 0
    
    print(f"Dynamically tuned max_array_chunk = {max_array_chunk}")
    
    # Start timing
    union_start = time.time()
    
    # Get more detailed statistics about array sizes
    size_stats = size_df.agg(
        expr("avg(array_size)").alias("mean_size"),
        expr("percentile(array_size, 0.95)").alias("p95_size"),
        expr("percentile(array_size, 0.99)").alias("p99_size"),
        expr("max(array_size)").alias("max_size")
    ).collect()[0]
    
    mean_size = size_stats["mean_size"]
    p95_size = size_stats["p95_size"] if p95 == 0 else p95  # Use previously calculated p95 if available
    p99_size = size_stats["p99_size"] if p99 == 0 else p99  # Use previously calculated p99 if available
    max_size = size_stats["max_size"]
    
    print(f"Array size statistics - Mean: {mean_size:.2f}, 95th percentile: {p95_size}, 99th percentile: {p99_size}, Max: {max_size}")
    
    # Define threshold for large arrays that need chunking
    # Arrays larger than this will be split into chunks
    large_threshold = max(p95_size, max_array_chunk)
    print(f"Using threshold of {large_threshold} for arrays that need chunking")
    
    # Create a temporary view for SQL operations
    df.createOrReplaceTempView("array_data")
    
    # Create two datasets:
    # 1. Small arrays (no chunking needed)
    # 2. Large arrays (need chunking)
    
    # First handle normal sized arrays with standard explode
    print("Processing normal sized arrays (no chunking needed)...")
    
    normal_sql = f"""
    SELECT * FROM array_data 
    WHERE size(items_array) <= {large_threshold}
    """
    
    normal_df = df.sparkSession.sql(normal_sql)
    normal_count = normal_df.count()
    print(f"Found {normal_count} records with normal sized arrays")
    
    if normal_count > 0:
        normal_exploded = normal_df.selectExpr(
            "record_id",
            "inline(items_array)"
        )
    else:
        # Create empty DataFrame with same schema
        normal_exploded = df.sparkSession.createDataFrame(
            [],
            df.selectExpr("record_id", "inline(items_array)").schema
        )

    # Now handle large arrays that need chunking
    print("Processing large arrays with chunking...")
    
    large_sql = f"""
    SELECT * FROM array_data 
    WHERE size(items_array) > {large_threshold}
    """
    
    large_df = df.sparkSession.sql(large_sql)
    large_count = large_df.count()
    print(f"Found {large_count} records with large arrays that need chunking")
    
    # If no large arrays, just return the normal results
    if large_count == 0:
        print("No large arrays found, using standard explode only")
        union_end = time.time()
        union_duration = union_end - union_start
        
        # Count to materialize
        final_count = normal_exploded.count()
        
        print(f"Union-based explosion completed in {union_duration:.2f} seconds, yielding {final_count} records")
        
        # Analyze partition distribution
        try:
            partition_counts = normal_exploded.groupBy(spark_partition_id()).count()
            partition_stats = partition_counts.agg(
                expr("stddev(count)").alias("stddev"),
                expr("avg(count)").alias("mean")
            ).collect()[0]
            union_cv = partition_stats["stddev"] / partition_stats["mean"] if partition_stats["mean"] > 0 else 0
            print(f"Partition balance (CV): {union_cv:.4f}")
        except Exception as e:
            print(f"Could not analyze partition distribution: {str(e)}")
            union_cv = float('inf')
        
        metrics = {
            "duration": union_duration,
            "count": final_count,
            "cv": union_cv
        }
        
        return normal_exploded, metrics
    
    # Process large arrays by splitting them into chunks
    print("Processing large arrays by splitting into chunks...")
    
    # Create a UDF to split arrays into chunks
    def split_array_to_chunks(record_id, items_array):
        if not items_array:
            return []
        
        result = []
        total_chunks = (len(items_array) + max_array_chunk - 1) // max_array_chunk
        
        for chunk_idx in range(total_chunks):
            start = chunk_idx * max_array_chunk
            end = min(start + max_array_chunk, len(items_array))
            chunk = items_array[start:end]
            
            # Create a row with record_id, chunk_idx, and the chunk
            result.append((record_id, chunk_idx, chunk))
        
        return result
    
    # Convert large_df to RDD for processing
    large_rdd = large_df.select("record_id", "items_array").rdd
    
    # Apply the chunking function
    chunked_rdd = large_rdd.flatMap(lambda row: split_array_to_chunks(row[0], row[1]))
    
    # Define the schema for the chunked data
    
    # Get the schema of the original items_array
    array_schema = large_df.schema["items_array"].dataType
    
    chunked_schema = StructType([
        StructField("record_id", IntegerType(), True),
        StructField("chunk_idx", IntegerType(), True),
        StructField("chunk_array", array_schema, True)
    ])
    
    # Convert back to DataFrame
    chunked_df = df.sparkSession.createDataFrame(chunked_rdd, chunked_schema)
    
    # Get some stats on the chunks
    chunk_stats = chunked_df.groupBy("record_id").agg(
        expr("count(*)").alias("num_chunks"),
        expr("sum(size(chunk_array))").alias("total_elements")
    )
    
    max_chunks = chunk_stats.agg(expr("max(num_chunks)")).collect()[0][0]
    avg_chunks = chunk_stats.agg(expr("avg(num_chunks)")).collect()[0][0]
    
    print(f"Maximum chunks per record: {max_chunks}")
    print(f"Average chunks per record: {avg_chunks:.2f}")
    
    # Calculate appropriate number of partitions for exploding the chunks
    # More chunks and larger arrays need more partitions
    chunk_count = chunked_df.count()
    partition_factor = chunk_count / normal_count if normal_count > 0 else 1
    chunk_partitions = max(20, int(partition_factor * 20))
    
    print(f"Using {chunk_partitions} partitions for chunk processing")
    
    # Redistribute chunks to partitions for better balance
    # Use both record_id and chunk_idx for distribution
    chunked_df = chunked_df.withColumn(
        "partition_key",
        expr("hash(concat(cast(record_id as string), '_', cast(chunk_idx as string)))")
    )
    
    chunked_df = chunked_df.repartition(chunk_partitions, "partition_key")
    
    # Now explode each chunk
    chunked_exploded = chunked_df.selectExpr(
        "record_id",
        "inline(chunk_array)"
    )
    
    # Union with the normal results
    all_exploded = normal_exploded.union(chunked_exploded)
    
    # Cache the result
    all_exploded.cache()
    
    # Count to materialize
    union_count = all_exploded.count()
    
    # Calculate duration
    union_end = time.time()
    union_duration = union_end - union_start
    
    print(f"Union-based explosion completed in {union_duration:.2f} seconds, yielding {union_count} records")
    
    # Analyze partition distribution
    try:
        print("\nAnalyzing partition distribution:")
        partition_counts = all_exploded.groupBy(spark_partition_id()).count()
        
        partition_stats = partition_counts.agg(
            expr("min(count)").alias("min"),
            expr("avg(count)").alias("mean"),
            expr("max(count)").alias("max"),
            expr("stddev(count)").alias("stddev")
        ).collect()[0]
        
        min_count = partition_stats["min"]
        mean_count = partition_stats["mean"]
        max_count = partition_stats["max"]
        stddev_count = partition_stats["stddev"]
        
        # Calculate coefficient of variation
        cv = stddev_count / mean_count if mean_count > 0 else 0
        
        print(f"Partition statistics:")
        print(f"- Min: {min_count}")
        print(f"- Mean: {mean_count:.2f}")
        print(f"- Max: {max_count}")
        print(f"- Stddev: {stddev_count:.2f}")
        print(f"- CV: {cv:.4f} (lower is better - indicates more balanced partitions)")
        
        # Calculate imbalance ratio
        imbalance = max_count / mean_count if mean_count > 0 else 0
        print(f"- Max/Mean ratio: {imbalance:.2f}x (lower is better)")
        
    except Exception as e:
        print(f"Could not analyze partition distribution: {str(e)}")
        cv = float('inf')
    
    # Return results
    metrics = {
        "duration": union_duration,
        "count": union_count,
        "cv": cv
    }
    
    return all_exploded, metrics

@log_timing("Flatmap explosion and analysis")
def flatmap_and_analyze(df):
    """Use flatMap transformation to process array data and analyze performance"""
    print("Processing array data using flatMap approach...")
    
    # First, let's analyze the array sizes before explosion
    print("Extracting array size information...")
    size_df = df.select(
        "record_id",
        expr("size(items_array)").alias("array_size")
    )
    
    # Analyze the array sizes before explosion
    size_stats = size_df.describe().collect()
    print("\nArray size statistics before explosion:")
    for row in size_stats:
        print(f"{row['summary']}: {row['array_size']}")
    
    # Calculate standard deviation and coefficient of variation for skew analysis
    std_stats = size_df.agg(
        expr("avg(array_size)").alias("mean_len"),
        expr("stddev(array_size)").alias("stddev_len")
    ).collect()[0]
    
    mean_len = std_stats["mean_len"]
    stddev_len = std_stats["stddev_len"]
    
    # Calculate coefficient of variation (CV) - normalized measure of dispersion
    cv = stddev_len / mean_len if mean_len > 0 else 0
    
    print(f"\nStandard deviation metrics:")
    print(f"Mean array length: {mean_len:.2f}")
    print(f"Standard deviation: {stddev_len:.2f}")
    print(f"Coefficient of variation: {cv:.4f} (higher values indicate greater relative variability)")
    
    # Start timing the flatMap approach
    print("\nRunning flatMap explosion...")
    flatmap_start = time.time()
    
    # Convert to RDD for flatMap operation
    rdd = df.select("record_id", "items_array").rdd
    
    # Define the flatMap function
    def flatten_array(row):
        record_id = row[0]
        items = row[1]
        results = []
        
        # Skip processing if items is None
        if not items:
            return results
            
        # Process each element in the array
        for item in items:
            # Extract fields from the struct
            element_id = item.element_id if hasattr(item, 'element_id') else None
            element_name = item.element_name if hasattr(item, 'element_name') else None
            element_value = item.element_value if hasattr(item, 'element_value') else None
            element_category = item.element_category if hasattr(item, 'element_category') else None
            element_active = item.element_active if hasattr(item, 'element_active') else None
            element_score = item.element_score if hasattr(item, 'element_score') else None
            
            # Create a new row
            results.append((
                record_id,
                element_id,
                element_name,
                element_value,
                element_category,
                element_active,
                element_score
            ))
        
        return results
    
    # Apply the flatMap transformation
    flattened_rdd = rdd.flatMap(flatten_array)
    
    # Convert back to DataFrame
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
    
    # Define the schema for the flattened data
    schema = StructType([
        StructField("record_id", IntegerType(), True),
        StructField("element_id", IntegerType(), True),
        StructField("element_name", StringType(), True),
        StructField("element_value", DoubleType(), True),
        StructField("element_category", StringType(), True),
        StructField("element_active", BooleanType(), True),
        StructField("element_score", DoubleType(), True)
    ])
    
    # Create DataFrame from RDD with the defined schema
    flattened_df = df.sparkSession.createDataFrame(flattened_rdd, schema)
    
    # Cache the result for further analysis
    flattened_df.cache()
    
    # Count the rows to materialize the cache
    flatmap_count = flattened_df.count()
    
    # Calculate duration
    flatmap_end = time.time()
    flatmap_duration = flatmap_end - flatmap_start
    
    print(f"FlatMap explosion completed in {flatmap_duration:.2f} seconds, yielding {flatmap_count} records")
    
    # Analyze record distribution
    try:
        print("\nAnalyzing records per source record:")
        record_counts = flattened_df.groupBy("record_id").count()
        record_summary = record_counts.summary("min", "25%", "50%", "75%", "max").collect()
        
        print("Elements per record statistics:")
        for row in record_summary:
            print(f"{row['summary']}: {row['count']}")
            
        # Get top records with most elements
        print("\nRecords with most elements:")
        record_counts.orderBy(col("count").desc()).show(10)
        
        # Calculate the skew factor if possible
        try:
            max_elements = float(record_summary[4]["count"])  # 'max' is at index 4
            avg_elements = record_counts.select(expr("avg(count)")).collect()[0][0]
            skew_factor = max_elements / avg_elements
            print(f"Skew factor (max/avg): {skew_factor:.2f}x")
        except Exception as skew_e:
            print(f"Could not calculate skew factor: {str(skew_e)}")
    except Exception as e:
        print(f"Could not analyze records distribution: {str(e)}")
    
    # Try to analyze partition distribution
    try:
        print("\nAnalyzing partition distribution:")
        partition_counts = flattened_df.groupBy(spark_partition_id()).count()
        partition_summary = partition_counts.summary("min", "25%", "50%", "75%", "max").collect()
        
        print("Partition size statistics:")
        for row in partition_summary:
            print(f"{row['summary']}: {row['count']}")
            
        # Get top 5 largest partitions
        print("\nLargest partitions:")
        partition_counts.orderBy(col("count").desc()).show(5)
        
        # Calculate coefficient of variation for partition sizes
        partition_stats = partition_counts.agg(
            expr("avg(count)").alias("mean_size"),
            expr("stddev(count)").alias("stddev_size")
        ).collect()[0]
        
        mean_size = partition_stats["mean_size"]
        stddev_size = partition_stats["stddev_size"]
        partition_cv = stddev_size / mean_size if mean_size > 0 else 0
        
        print(f"Partition size variation: {partition_cv:.4f} (lower is better - indicates more balanced partitions)")
    except Exception as e:
        print(f"Could not analyze partition distribution: {str(e)}")
    
    # Return the results
    flatmap_metrics = {
        "duration": flatmap_duration,
        "count": flatmap_count
    }
    
    return flattened_df, flatmap_metrics

@log_timing("Comprehensive benchmark of all approaches")
def comprehensive_benchmark(df, num_partitions=10):
    """
    Run a comprehensive benchmark of all approaches for handling skewed arrays
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with the array column to explode
    num_partitions : int
        Number of partitions to use for approaches that need it
        
    Returns:
    --------
    dict
        Results and metrics for all approaches
    """
    # Import necessary functions
    import math
    from pyspark.sql.functions import udf, col, expr, spark_partition_id, lit, when
    from pyspark.sql.types import IntegerType
    
    # Initialize result dictionary with default values
    # This ensures the function always returns a dictionary even if errors occur
    result_dict = {
        "explode": {
            "duration": 0,
            "count": 0,
            "cv": float('inf'),
            "df": None
        },
        "salting": {
            "duration": 0,
            "count": 0,
            "cv": float('inf'),
            "df": None
        },
        "bucketing": {
            "duration": 0,
            "count": 0,
            "cv": float('inf'),
            "df": None
        },
        "union": {
            "duration": 0,
            "count": 0,
            "cv": float('inf'),
            "df": None
        },
        "reshuffle": {
            "duration": 0,
            "count": 0,
            "cv": float('inf'),
            "df": None
        },
        "flatmap": {
            "duration": 0,
            "count": 0,
            "cv": float('inf'),
            "df": None
        },
        "time_winner": "N/A",
        "balance_winner": "N/A"
    }
    
    print("======== COMPREHENSIVE BENCHMARK OF ALL APPROACHES (FIXED) ========")
    
    try:
        # 1. Standard explode (baseline)
        print("\n=== APPROACH 1: STANDARD EXPLODE (BASELINE) ===")
        explode_start = time.time()
        
        try:
            exploded_df = df.selectExpr(
                "record_id",
                "inline(items_array)"
            ).select(
                "record_id",
                "element_id",
                "element_category",
                "element_score",
                "element_value",
                "element_name",
                "element_active"
            )
            
            # Cache and count
            exploded_df.cache()
            explode_count = exploded_df.count()
            explode_end = time.time()
            explode_duration = explode_end - explode_start
            
            print(f"Standard explode completed in {explode_duration:.2f} seconds, yielding {explode_count} records")
            
            # Analyze partition distribution
            try:
                explode_parts = exploded_df.groupBy(spark_partition_id()).count()
                explode_stats = explode_parts.agg(
                    expr("stddev(count)").alias("stddev"),
                    expr("avg(count)").alias("mean")
                ).collect()[0]
                explode_cv = explode_stats["stddev"] / explode_stats["mean"] if explode_stats["mean"] > 0 else 0
                print(f"Partition balance (CV): {explode_cv:.4f}")
            except Exception as e:
                print(f"Could not analyze partition distribution: {str(e)}")
                explode_cv = float('inf')
            
            # Update result dictionary
            result_dict["explode"]["duration"] = explode_duration
            result_dict["explode"]["count"] = explode_count
            result_dict["explode"]["cv"] = explode_cv
            result_dict["explode"]["df"] = exploded_df
            
        except Exception as e:
            print(f"Error during standard explode: {str(e)}")
            explode_duration = float('inf')
            explode_count = 0
            explode_cv = float('inf')
        
        # 2. Salting approach
        print("\n=== APPROACH 2: SALTED ARRAY EXPLOSION ===")
        try:
            salted_df, salting_metrics = salted_array_explosion(df, num_partitions)
            salting_duration = salting_metrics["duration"]
            salting_count = salting_metrics["count"]
            salting_cv = salting_metrics["cv"]
            
            # Select the same columns for fair comparison
            salted_df = salted_df.select(
                "record_id",
                "element_id",
                "element_category",
                "element_score",
                "element_value",
                "element_name",
                "element_active"
            )
            
            # Update result dictionary
            result_dict["salting"]["duration"] = salting_duration
            result_dict["salting"]["count"] = salting_count
            result_dict["salting"]["cv"] = salting_cv
            result_dict["salting"]["df"] = salted_df
            
        except Exception as e:
            print(f"Error during salted explosion: {str(e)}")
            salting_duration = float('inf')
            salting_count = 0
            salting_cv = float('inf')
        
        # 3. Bucketing approach
        print("\n=== APPROACH 3: BUCKETED ARRAY EXPLOSION ===")
        try:
            bucket_df, bucket_metrics = bucketed_array_explosion(df, num_partitions)
            bucket_duration = bucket_metrics["duration"]
            bucket_count = bucket_metrics["count"]
            bucket_cv = bucket_metrics["cv"]
            
            # Select the same columns for fair comparison
            bucket_df = bucket_df.select(
                "record_id",
                "element_id",
                "element_category",
                "element_score",
                "element_value",
                "element_name",
                "element_active"
            )
            
            # Update result dictionary
            result_dict["bucketing"]["duration"] = bucket_duration
            result_dict["bucketing"]["count"] = bucket_count
            result_dict["bucketing"]["cv"] = bucket_cv
            result_dict["bucketing"]["df"] = bucket_df
            
        except Exception as e:
            print(f"Error during bucketed explosion: {str(e)}")
            bucket_duration = float('inf')
            bucket_count = 0
            bucket_cv = float('inf')
        
        # 4. Union-based approach
        print("\n=== APPROACH 4: UNION-BASED ARRAY EXPLOSION ===")
        try:
            union_df, union_metrics = union_based_explosion(df)  # Set chunk size to 500
            union_duration = union_metrics["duration"]
            union_count = union_metrics["count"]
            union_cv = union_metrics["cv"]
            
            # Select the same columns for fair comparison
            union_df = union_df.select(
                "record_id",
                "element_id",
                "element_category",
                "element_score",
                "element_value",
                "element_name",
                "element_active"
            )
            
            # Update result dictionary
            result_dict["union"]["duration"] = union_duration
            result_dict["union"]["count"] = union_count
            result_dict["union"]["cv"] = union_cv
            result_dict["union"]["df"] = union_df
            
        except Exception as e:
            print(f"Error during union-based explosion: {str(e)}")
            union_duration = float('inf')
            union_count = 0
            union_cv = float('inf')
        
        # 5. Reshuffle approach
        print("\n=== APPROACH 5: RESHUFFLE THEN EXPLODE ===")
        try:
            reshuffle_start = time.time()
            
            # Add column with array size
            df_with_size = df.withColumn("array_size", expr("size(items_array)"))
            
            # Calculate statistics for balancing
            size_stats = df_with_size.agg(
                expr("avg(array_size)").alias("avg_size"),
                expr("percentile(array_size, 0.95)").alias("p95_size")
            ).collect()[0]
            
            avg_size = size_stats["avg_size"]
            p95_size = size_stats["p95_size"]
            
            # Define a weight factor for each record based on its array size
            df_with_weight = df_with_size.withColumn(
                "distribution_weight", 
                when(col("array_size") > lit(p95_size), 
                     # Records with very large arrays get more weight
                     expr(f"ceil(array_size / greatest(1.0, {avg_size} / 10))")
                ).otherwise(
                     # Normal records get weight proportional to array size
                     expr(f"ceil(array_size / greatest(1.0, {avg_size} / 2))")
                )
            )
            
            # Calculate a distribution key that will be used for partitioning
            df_with_key = df_with_weight.withColumn(
                "distribution_key",
                # Hash combination of record_id and a modulo based on weight
                expr("hash(concat(cast(record_id as string), cast(pmod(record_id, distribution_weight) as string)))")
            )
            
            # Repartition based on distribution key to spread the workload
            reshuffled_df = df_with_key.repartition(num_partitions, "distribution_key")
            
            # Explode after reshuffling
            reshuffle_exploded = reshuffled_df.selectExpr(
                "record_id",
                "inline(items_array)"
            ).select(
                "record_id",
                "element_id",
                "element_category",
                "element_score",
                "element_value",
                "element_name",
                "element_active"
            )
            
            # Cache and count
            reshuffle_exploded.cache()
            reshuffle_count = reshuffle_exploded.count()
            reshuffle_end = time.time()
            reshuffle_duration = reshuffle_end - reshuffle_start
            
            print(f"Reshuffle+explode completed in {reshuffle_duration:.2f} seconds, yielding {reshuffle_count} records")
            
            # Analyze partition distribution for reshuffle+explode
            try:
                reshuffle_parts = reshuffle_exploded.groupBy(spark_partition_id()).count()
                reshuffle_stats = reshuffle_parts.agg(
                    expr("stddev(count)").alias("stddev"),
                    expr("avg(count)").alias("mean")
                ).collect()[0]
                reshuffle_cv = reshuffle_stats["stddev"] / reshuffle_stats["mean"] if reshuffle_stats["mean"] > 0 else 0
                print(f"Partition balance (CV): {reshuffle_cv:.4f}")
            except Exception as e:
                print(f"Could not analyze partition distribution: {str(e)}")
                reshuffle_cv = float('inf')
            
            # Update result dictionary
            result_dict["reshuffle"]["duration"] = reshuffle_duration
            result_dict["reshuffle"]["count"] = reshuffle_count
            result_dict["reshuffle"]["cv"] = reshuffle_cv
            result_dict["reshuffle"]["df"] = reshuffle_exploded
            
        except Exception as e:
            print(f"Error during reshuffle+explode: {str(e)}")
            reshuffle_duration = float('inf')
            reshuffle_count = 0
            reshuffle_cv = float('inf')
            
        # 6. FlatMap approach
        print("\n=== APPROACH 6: FLATMAP WITH SKEW HANDLING ===")
        try:
            flatmap_df, flatmap_metrics = flatmap_with_skew_handling(df)
            flatmap_duration = flatmap_metrics["duration"]
            flatmap_count = flatmap_metrics["count"]
            
            # Analyze partition distribution for flatmap
            try:
                flatmap_parts = flatmap_df.groupBy(spark_partition_id()).count()
                flatmap_stats = flatmap_parts.agg(
                    expr("stddev(count)").alias("stddev"),
                    expr("avg(count)").alias("mean")
                ).collect()[0]
                flatmap_cv = flatmap_stats["stddev"] / flatmap_stats["mean"] if flatmap_stats["mean"] > 0 else 0
                print(f"Partition balance (CV): {flatmap_cv:.4f}")
            except Exception as e:
                print(f"Could not analyze partition distribution: {str(e)}")
                flatmap_cv = float('inf')
            
            # Update result dictionary
            result_dict["flatmap"]["duration"] = flatmap_duration
            result_dict["flatmap"]["count"] = flatmap_count
            result_dict["flatmap"]["cv"] = flatmap_cv
            result_dict["flatmap"]["df"] = flatmap_df
            
        except Exception as e:
            print(f"Error during flatmap with skew handling: {str(e)}")
            flatmap_duration = float('inf')
            flatmap_count = 0
            flatmap_cv = float('inf')
        
        # Compare all approaches
        print("\n=== PERFORMANCE COMPARISON SUMMARY ===")
        
        # Get all durations
        all_durations = [
            result_dict["explode"]["duration"],
            result_dict["salting"]["duration"], 
            result_dict["bucketing"]["duration"],
            result_dict["union"]["duration"], 
            result_dict["reshuffle"]["duration"], 
            result_dict["flatmap"]["duration"]
        ]
        
        # Filter out infinity values
        valid_durations = [d for d in all_durations if d != float('inf')]
        
        if valid_durations:
            fastest_time = min(valid_durations)
            
            # Calculate ratios
            explode_ratio = result_dict["explode"]["duration"] / fastest_time if result_dict["explode"]["duration"] != float('inf') else float('inf')
            salting_ratio = result_dict["salting"]["duration"] / fastest_time if result_dict["salting"]["duration"] != float('inf') else float('inf')
            bucket_ratio = result_dict["bucketing"]["duration"] / fastest_time if result_dict["bucketing"]["duration"] != float('inf') else float('inf')
            union_ratio = result_dict["union"]["duration"] / fastest_time if result_dict["union"]["duration"] != float('inf') else float('inf')
            reshuffle_ratio = result_dict["reshuffle"]["duration"] / fastest_time if result_dict["reshuffle"]["duration"] != float('inf') else float('inf')
            flatmap_ratio = result_dict["flatmap"]["duration"] / fastest_time if result_dict["flatmap"]["duration"] != float('inf') else float('inf')
            
            # Print performance summary
            print(f"1. Standard explode:     {result_dict['explode']['duration']:.2f} s for {result_dict['explode']['count']} records")
            print(f"2. Salted explosion:     {result_dict['salting']['duration']:.2f} s for {result_dict['salting']['count']} records")
            print(f"3. Bucketed explosion:   {result_dict['bucketing']['duration']:.2f} s for {result_dict['bucketing']['count']} records")
            print(f"4. Union-based:          {result_dict['union']['duration']:.2f} s for {result_dict['union']['count']} records")
            print(f"5. Reshuffle+explode:    {result_dict['reshuffle']['duration']:.2f} s for {result_dict['reshuffle']['count']} records")
            print(f"6. FlatMap with skew:    {result_dict['flatmap']['duration']:.2f} s for {result_dict['flatmap']['count']} records")
            
            print(f"\nExecution time ratios (lower is better):")
            print(f"1. Standard explode:     {explode_ratio:.2f}x")
            print(f"2. Salted explosion:     {salting_ratio:.2f}x")
            print(f"3. Bucketed explosion:   {bucket_ratio:.2f}x")
            print(f"4. Union-based:          {union_ratio:.2f}x")
            print(f"5. Reshuffle+explode:    {reshuffle_ratio:.2f}x")
            print(f"6. FlatMap with skew:    {flatmap_ratio:.2f}x")
            
            # Determine the winner based on execution time
            approach_names = ["Standard explode", "Salted explosion", "Bucketed explosion", 
                            "Union-based explosion", "Reshuffle+explode", "FlatMap with skew handling"]
            
            valid_indices = [i for i, d in enumerate(all_durations) if d != float('inf')]
            if valid_indices:
                time_winner_idx = valid_indices[all_durations.index(fastest_time)]
                time_winner = approach_names[time_winner_idx]
                result_dict["time_winner"] = time_winner
                print(f"\nFastest approach: {time_winner} ({fastest_time:.2f} seconds)")
            
            # Compare partition balance
            print("\n=== PARTITION BALANCE COMPARISON ===")
            print(f"1. Standard explode:     CV = {result_dict['explode']['cv']:.4f}")
            print(f"2. Salted explosion:     CV = {result_dict['salting']['cv']:.4f}")
            print(f"3. Bucketed explosion:   CV = {result_dict['bucketing']['cv']:.4f}")
            print(f"4. Union-based:          CV = {result_dict['union']['cv']:.4f}")
            print(f"5. Reshuffle+explode:    CV = {result_dict['reshuffle']['cv']:.4f}")
            print(f"6. FlatMap with skew:    CV = {result_dict['flatmap']['cv']:.4f}")
            
            # Determine which has the most balanced partitions
            all_cvs = [
                result_dict["explode"]["cv"],
                result_dict["salting"]["cv"], 
                result_dict["bucketing"]["cv"],
                result_dict["union"]["cv"], 
                result_dict["reshuffle"]["cv"], 
                result_dict["flatmap"]["cv"]
            ]
            
            valid_cvs = [cv for cv in all_cvs if cv != float('inf')]
            if valid_cvs:
                best_cv = min(valid_cvs)
                valid_cv_indices = [i for i, cv in enumerate(all_cvs) if cv != float('inf')]
                if valid_cv_indices:
                    balance_winner_idx = valid_cv_indices[all_cvs.index(best_cv)]
                    balance_winner = approach_names[balance_winner_idx]
                    result_dict["balance_winner"] = balance_winner
                    print(f"Most balanced approach: {balance_winner} (CV = {best_cv:.4f})")
        
    except Exception as e:
        print(f"Error during benchmark comparison: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Debug confirmation
    print("\nDebug: Returning result dictionary")
    
    # Print diagnostic info about the result dictionary
    print(f"Result dict has {len(result_dict)} entries")
    print(f"Keys: {', '.join(result_dict.keys())}")
    
    # Always return the result dictionary, even if errors occurred
    return result_dict


@log_timing("Full skewed array benchmark")
def run_skewed_array_benchmark():
    """Run a complete benchmark of all approaches with genuinely skewed data"""
    print("======== STARTING SKEWED ARRAY BENCHMARK WITH GENUINE SKEW ========")
    
    # Create Spark session with improved memory settings
    spark = SparkSession.builder \
            .appName("Skewed Dataset Benchmark") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .config("spark.cleaner.periodicGC.interval", "15min") \
            .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
            .getOrCreate()
    
    try:
        # Generate base dataset with SKEWED array field
        print("\nGenerating dataset with genuinely skewed array distribution...")
        
        # Use a smaller record count for testing
        test_record_count = 100000  # Reduced for testing
        
        # Generate the dataset
        base_df = generate_base_dataset(spark, test_record_count)
        
        # Verify schema
        print("\nDataset schema:")
        base_df.printSchema()
        
        # Verify skew in the array sizes
        print("\nVerifying array skew:")
        size_df = base_df.select(
            "record_id",
            expr("size(items_array)").alias("array_size")
        )
        
        # Show distribution
        print("\nArray size distribution statistics:")
        size_df.describe().show()
        
        # Show histogram 
        print("\nArray size histogram (sample):")
        size_df.groupBy("array_size").count().orderBy("array_size").show(20)
        
        # Run comprehensive benchmark with all approaches
        print("\nRunning comprehensive benchmark with all approaches...")
        benchmark_results = comprehensive_benchmark(base_df)
        
        # Output the final summary
        print("\n======== SKEWED ARRAY BENCHMARK SUMMARY ========")
        print(f"Dataset size: {test_record_count} base records")
        
        # Compare record counts
        print(f"\nRecord counts after processing:")
        print(f"- Standard explode:    {benchmark_results['explode']['count']:,}")
        print(f"- Salted explosion:    {benchmark_results['salting']['count']:,}")
        print(f"- Bucketed explosion:  {benchmark_results['bucketing']['count']:,}")
        print(f"- Union-based:         {benchmark_results['union']['count']:,}")
        print(f"- Reshuffle+explode:   {benchmark_results['reshuffle']['count']:,}")
        print(f"- FlatMap with skew:   {benchmark_results['flatmap']['count']:,}")
        
        # Compare execution times
        print(f"\nExecution times:")
        print(f"- Standard explode:    {benchmark_results['explode']['duration']:.2f} seconds")
        print(f"- Salted explosion:    {benchmark_results['salting']['duration']:.2f} seconds")
        print(f"- Bucketed explosion:  {benchmark_results['bucketing']['duration']:.2f} seconds")
        print(f"- Union-based:         {benchmark_results['union']['duration']:.2f} seconds")
        print(f"- Reshuffle+explode:   {benchmark_results['reshuffle']['duration']:.2f} seconds")
        print(f"- FlatMap with skew:   {benchmark_results['flatmap']['duration']:.2f} seconds")
        
        # Compare partition balance
        print(f"\nPartition balance (coefficient of variation - lower is better):")
        print(f"- Standard explode:    {benchmark_results['explode']['cv']:.4f}")
        print(f"- Salted explosion:    {benchmark_results['salting']['cv']:.4f}")
        print(f"- Bucketed explosion:  {benchmark_results['bucketing']['cv']:.4f}")
        print(f"- Union-based:         {benchmark_results['union']['cv']:.4f}")
        print(f"- Reshuffle+explode:   {benchmark_results['reshuffle']['cv']:.4f}")
        print(f"- FlatMap with skew:   {benchmark_results['flatmap']['cv']:.4f}")
        
        # Overall winner
        print(f"\nOverall recommendations:")
        print(f"- Fastest approach: {benchmark_results['time_winner']}")
        print(f"- Most balanced approach: {benchmark_results['balance_winner']}")
        
        print("\n======== SKEWED ARRAY BENCHMARK COMPLETE ========")
        
        return benchmark_results
        
    except Exception as e:
        print(f"ERROR in benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    run_skewed_array_benchmark()
