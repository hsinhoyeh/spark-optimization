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
    when, broadcast, count, sum, avg
)

# Configuration
NUM_RECORDS = 1_000_000  # Base number of records
NUM_FIELDS = 500        # Number of fields per record
ARRAY_MIN_SIZE = 1000      # Minimum array size
ARRAY_MAX_SIZE = 100000    # Maximum array size
SKEW_PERCENTAGE = 0.3   # Percentage of records with skewed data

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
            
            # Shuffle partitions based on data size
            .config("spark.sql.shuffle.partitions", "200")
            
            # Memory and execution configurations
            .config("spark.executor.memory", "4g")
            .config("spark.driver.memory", "4g")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "2g")
            
            # Optimized I/O
            .config("spark.io.compression.codec", "lz4")
            .config("spark.reducer.maxSizeInFlight", "96m")
            
            # Fault tolerance
            .config("spark.task.maxFailures", "5")
            
            .getOrCreate())

@log_timing("Generate skewed base dataset")
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
    
    # Add the array field that we'll later explode
    # This is intentionally skewed - some records have many array elements, others have few
    df = df.withColumn("array_sizes", 
        when(rand() < SKEW_PERCENTAGE,
             # Skewed records have large arrays (up to ARRAY_MAX_SIZE elements)
             (rand() * (ARRAY_MAX_SIZE - ARRAY_MIN_SIZE) + ARRAY_MIN_SIZE).cast("int")
        ).otherwise(
             # Non-skewed records have smaller arrays
             (rand() * 10 + ARRAY_MIN_SIZE).cast("int")
        ))
    
    # Define a UDF to create arrays of varying sizes with the skew we want
    @log_timing("Generate array field")
    def generate_array_field(df):
        """Generate the array field with skewed distribution"""
        # We'll use SQL expressions to build arrays of dynamic size
        # This approach is more efficient than UDFs for large-scale operations
        
        # First create a column with array element templates
        array_expr = "array("
        for i in range(ARRAY_MAX_SIZE):
            # Each array element is a struct with multiple fields
            # to simulate complex nested data
            if i > 0:
                array_expr += ","
            array_expr += f"""
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
                                         concat('tag_', cast((id % 10) as string)),
                                         concat('tag_', cast((id % 7 + 10) as string)),
                                         concat('tag_', cast((id % 5 + 20) as string))
                                       ),
                        'element_active', (id + {i}) % 2 = 0,
                        'element_score', rand() * 1000,
                        'element_date', date_add(to_date('2023-01-01'), cast(rand() * 365 as int))
                    )
                else
                    null
                end"""
        array_expr += ")"
        
        # Apply expression to create the array field, filtering out null values
        return df.withColumn("items_array_raw", expr(array_expr))\
                 .withColumn("items_array", 
                             expr("filter(items_array_raw, x -> x is not null)"))
    
    # Generate the array field
    df = generate_array_field(df)
    
    # Drop intermediate columns
    df = df.drop("array_sizes", "items_array_raw", "salt", "id")
    
    return df

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

@log_timing("Explode array and analyze skew")
def explode_and_analyze(df):
    """Explode the array field and analyze the resulting skew"""
    print("Exploding the array field...")
    
    # Explode the array field
    exploded_df = df.select(
        "record_id", 
        "creation_time",
        # Include some representative fields from the base record
        "string_field_1", 
        "int_field_1",
        "double_field_1",
        "key_field_1",
        # Explode the array field
        explode(col("items_array")).alias("item")
    )
    
    # Create final exploded view with flattened item fields
    exploded_df = exploded_df.select(
        "record_id",
        "creation_time", 
        "string_field_1", 
        "int_field_1",
        "double_field_1",
        "key_field_1",
        col("item.element_id").alias("element_id"),
        col("item.element_name").alias("element_name"),
        col("item.element_value").alias("element_value"),
        col("item.element_category").alias("element_category"),
        col("item.element_tags").alias("element_tags"),
        col("item.element_active").alias("element_active"),
        col("item.element_score").alias("element_score"),
        col("item.element_date").alias("element_date")
    )
    
    # Cache the result for multiple analyses
    exploded_df.cache()
    
    # Analyze the skew
    print("\nAnalyzing data skew after explosion...")
    
    # Overall row count after explosion
    total_rows = exploded_df.count()
    print(f"Total rows after explosion: {total_rows}")
    
    # Analyze distribution by record_id to see the skew
    record_distribution = exploded_df.groupBy("record_id")\
                                     .agg(count("*").alias("elements_per_record"))\
                                     .orderBy(col("elements_per_record").desc())
    
    # Get the top 10 records by array size
    top_records = record_distribution.limit(10).collect()
    print("\nTop 10 records by array size:")
    for row in top_records:
        print(f"Record ID: {row['record_id']}, Elements: {row['elements_per_record']}")
    
    # Get some statistics about the distribution
    distribution_stats = record_distribution.select(
        avg("elements_per_record").alias("avg_elements"),
        expr("percentile(elements_per_record, 0.5)").alias("median_elements"),
        expr("percentile(elements_per_record, 0.95)").alias("p95_elements"),
        expr("percentile(elements_per_record, 0.99)").alias("p99_elements"),
        expr("max(elements_per_record)").alias("max_elements")
    ).collect()[0]
    
    print("\nArray size distribution statistics:")
    print(f"Average elements per record: {distribution_stats['avg_elements']:.2f}")
    print(f"Median elements per record: {distribution_stats['median_elements']}")
    print(f"95th percentile: {distribution_stats['p95_elements']} elements")
    print(f"99th percentile: {distribution_stats['p99_elements']} elements")
    print(f"Maximum elements: {distribution_stats['max_elements']}")
    
    # Calculate skew factor (ratio of max to average)
    skew_factor = distribution_stats['max_elements'] / distribution_stats['avg_elements']
    print(f"Skew factor (max/avg): {skew_factor:.2f}x")
    
    # Analyze category distribution
    category_distribution = exploded_df.groupBy("element_category")\
                                       .agg(count("*").alias("count"))\
                                       .orderBy(col("count").desc())
    
    print("\nCategory distribution:")
    for row in category_distribution.collect():
        pct = 100.0 * row['count'] / total_rows
        print(f"Category: {row['element_category']}, Count: {row['count']} ({pct:.2f}%)")
    
    return exploded_df

@log_timing("Perform complex queries with skew handling")
def perform_complex_queries(df, exploded_df):
    """Perform complex queries on the exploded data with skew handling techniques"""
    print("Performing complex queries with skew handling techniques...")
    
    # Register temporary views for SQL
    df.createOrReplaceTempView("base_records")
    exploded_df.createOrReplaceTempView("exploded_items")
    
    spark = df.sparkSession
    
    # Example 1: Aggregation query with skew handling
    print("\nPerforming aggregation query with skew handling...")
    
    # Use salting technique to distribute skewed keys better
    # This is just for demonstration - in a real scenario,
    # you would apply more sophisticated techniques
    
    # Add a salt column to the exploded_items view
    spark.sql("""
    CREATE OR REPLACE TEMPORARY VIEW salted_items AS
    SELECT 
        *, 
        CAST(RAND() * 10 AS INT) AS salt
    FROM exploded_items
    """)
    
    # Perform a query that would typically suffer from skew
    # Using the salt to distribute processing
    agg_result = spark.sql("""
    SELECT 
        element_category,
        key_field_1 % 10 AS key_group,
        COUNT(*) AS item_count,
        AVG(element_score) AS avg_score,
        SUM(element_value) AS total_value
    FROM salted_items
    GROUP BY element_category, key_field_1 % 10, salt
    """)
    
    # Combine the salted results
    final_agg = spark.sql("""
    SELECT 
        element_category,
        key_group,
        SUM(item_count) AS item_count,
        AVG(avg_score) AS avg_score,
        SUM(total_value) AS total_value
    FROM (
        SELECT 
            element_category,
            key_group,
            item_count,
            avg_score,
            total_value
        FROM agg_result
    )
    GROUP BY element_category, key_group
    ORDER BY item_count DESC
    """)
    
    # Display results
    print("\nAggregation results (top 5):")
    final_agg.show(5)
    
    # Example 2: Join query with skew handling
    print("\nPerforming join with skew handling techniques...")
    
    # Create a small lookup table
    lookup_data = [(i, f"Category {chr(65+i%4)}", random.random() * 100) 
                  for i in range(10)]
    lookup_df = spark.createDataFrame(
        lookup_data, 
        ["category_id", "category_name", "category_weight"]
    )
    lookup_df.createOrReplaceTempView("category_lookup")
    
    # Broadcast the small table to optimize the join
    # Use the BROADCAST hint with the /*+ */ syntax
    join_result = spark.sql("""
    SELECT /*+ BROADCAST(cl) */
        e.record_id,
        e.element_id,
        e.element_name,
        e.element_category,
        cl.category_weight,
        e.element_score * cl.category_weight AS weighted_score
    FROM exploded_items e
    JOIN category_lookup cl
        ON e.element_category = concat('Category ', cl.category_name)
    WHERE e.element_active = true
    """)
    
    # Display results
    print("\nJoin results (top 5):")
    join_result.show(5)
    
    # Example 3: Window function over skewed data
    # By partitioning properly, we can handle the skew
    print("\nPerforming window functions with skew handling...")
    
    window_result = spark.sql("""
    SELECT
        record_id,
        element_id,
        element_name,
        element_score,
        element_category,
        RANK() OVER (PARTITION BY record_id ORDER BY element_score DESC) AS score_rank_in_record,
        element_score / SUM(element_score) OVER (PARTITION BY record_id) AS score_pct_of_record,
        AVG(element_score) OVER (PARTITION BY element_category) AS category_avg_score
    FROM exploded_items
    """)
    
    # Add a filter to see just the top scores per record
    top_scores = spark.sql("""
    SELECT * FROM window_result
    WHERE score_rank_in_record <= 3
    ORDER BY record_id, score_rank_in_record
    """)
    
    # Display results
    print("\nTop scores per record (sample):")
    top_scores.show(5)
    
    return {
        "aggregation": final_agg,
        "join": join_result,
        "window": top_scores
    }

@log_timing("Full skewed array dataset demonstration")
def run_skewed_array_demo():
    """Main function to demonstrate handling of skewed array data"""
    print("======== STARTING SKEWED ARRAY DATASET DEMONSTRATION ========")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Generate base dataset with hundreds of fields including the array field
        base_df = generate_base_dataset(spark)
        
        # Show the schema to verify we have the expected number of fields
        print("\nDataset schema:")
        base_df.printSchema()
        
        # Check a sample record
        print("\nSample record (truncated):")
        base_df.select("record_id", "creation_time", "string_field_1", "int_field_1", 
                      "double_field_1", "bool_field_1", "key_field_1", 
                      "items_array").show(1, truncate=True)
        
        # Count the base records
        base_count = base_df.count()
        print(f"\nBase dataset contains {base_count} records")
        
        # Check the array sizes
        print("\nAnalyzing array field sizes before explosion...")
        base_df.select(
            "record_id",
            expr("size(items_array)").alias("array_size")
        ).summary("min", "25%", "50%", "75%", "max").show()
        
        # Optional: Write to storage
        output_path = "./skewed_array_data"
        write_dataset(base_df, output_path)
        
        # Optional: Read from storage (to simulate a real-world scenario)
        # base_df = read_dataset(spark, output_path)
        
        # Explode the array and analyze the resulting skew
        exploded_df = explode_and_analyze(base_df)
        
        # Perform complex queries with skew handling techniques
        query_results = perform_complex_queries(base_df, exploded_df)
        
        print("\n======== SKEWED ARRAY DATASET DEMONSTRATION COMPLETE ========")
        
    except Exception as e:
        print(f"ERROR in demonstration: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("Stopping Spark session...")
        spark.stop()

if __name__ == "__main__":
    run_skewed_array_demo()
