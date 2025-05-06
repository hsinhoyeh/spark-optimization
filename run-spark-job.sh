#!/bin/bash
# run-spark-job.sh - Helper script for running Spark jobs via kubectl exec

# Default values
MASTER_POD="spark-master-0"
JOB_NAME="spark-job"
DRIVER_MEMORY="8g"
EXECUTOR_NUMBER=4
EXECUTOR_MEMORY="32g"
EXECUTOR_CORES="15"
JOB_FILE="pyscript/sheffle.py"
#JOB_FILE="pyscript/minimal_spark_test.py"
NVME_PATHS="/spark/nvme1/tmp,/spark/nvme2/tmp"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --pod)
      MASTER_POD="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --driver-memory)
      DRIVER_MEMORY="$2"
      shift 2
      ;;
    --run-executor)
      EXECUTOR_NUMBER="$2"
      shift 2
      ;;
    --executor-memory)
      EXECUTOR_MEMORY="$2"
      shift 2
      ;;
    --executor-cores)
      EXECUTOR_CORES="$2"
      shift 2
      ;;
    --job-file)
      JOB_FILE="$2"
      shift 2
      ;;
    --nvme-paths)
      NVME_PATHS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required parameters
if [ -z "$JOB_FILE" ]; then
  echo "Error: Job file path is required (--job-file)"
  echo "Usage: $0 --job-file /path/to/local/job.py [options]"
  exit 1
fi

# Get just the filename from the path
JOB_FILENAME=$(basename "$JOB_FILE")

echo "== Spark Job Execution via kubectl =="
echo "Master pod: $MASTER_POD"
echo "Job name: $JOB_NAME"
echo "Job file: $JOB_FILE"
echo "NVMe paths: $NVME_PATHS"

# Copy the job file to the pod
echo -e "\n== Copying job file to the pod =="
kubectl cp "$JOB_FILE" "$MASTER_POD:/tmp/$JOB_FILENAME"
if [ $? -ne 0 ]; then
  echo "Error: Failed to copy job file to the pod"
  exit 1
fi
echo "Job file copied successfully to /tmp/$JOB_FILENAME"

# Run the Spark job
echo -e "\n== Submitting Spark job =="
kubectl exec -it "$MASTER_POD" -- bash -c "/opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  --name \"$JOB_NAME\" \
  --driver-memory $DRIVER_MEMORY \
  --num-executors $EXECUTOR_NUMBER \
  --executor-memory $EXECUTOR_MEMORY \
  --executor-cores $EXECUTOR_CORES \
  --conf \"spark.local.dir=$NVME_PATHS\" \
  --conf \"spark.locality.wait=100ms\" \
  --conf \"spark.locality.wait.node=300ms\" \
  --conf \"spark.locality.wait.process=1s\" \
  --conf \"spark.shuffle.file.buffer=64k\" \
  --conf \"spark.io.compression.lz4.blockSize=512k\" \
  --conf \"spark.sql.adaptive.enabled=true\" \
  --conf \"spark.dynamicAllocation.enabled=false\" \
  /tmp/$JOB_FILENAME"

# Check result
if [ $? -eq 0 ]; then
  echo -e "\n== Job submitted successfully =="
else
  echo -e "\n== Job submission failed =="
fi
