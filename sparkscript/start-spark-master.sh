#!/bin/bash
# Script to start Spark master with proper network binding

# Get the IP address that we should bind to
POD_IP=$(hostname -i)
echo "Pod IP: $POD_IP"

# Check current environment
echo "Checking environment..."
echo "Current directory: $(pwd)"
echo "PATH: $PATH"
echo "SPARK_HOME: $SPARK_HOME"

# Try to locate spark-class
if [ -z "$SPARK_HOME" ]; then
  # Try to find SPARK_HOME
  echo "SPARK_HOME not set, attempting to find Spark installation..."
  if [ -d "/opt/spark" ]; then
    SPARK_HOME="/opt/spark"
    echo "Found Spark at $SPARK_HOME"
  elif [ -d "/usr/local/spark" ]; then
    SPARK_HOME="/usr/local/spark"
    echo "Found Spark at $SPARK_HOME"
  else
    # Find by looking for spark-class
    SPARK_CLASS_PATH=$(find / -name spark-class -type f 2>/dev/null | head -1)
    if [ -n "$SPARK_CLASS_PATH" ]; then
      SPARK_HOME=$(dirname $(dirname $SPARK_CLASS_PATH))
      echo "Found Spark at $SPARK_HOME based on spark-class location"
    else
      echo "ERROR: Could not locate Spark installation. Please set SPARK_HOME manually."
      exit 1
    fi
  fi
fi

# Ensure SPARK_HOME is in PATH
export PATH=$SPARK_HOME/bin:$PATH
echo "Updated PATH: $PATH"

# Check if something is already using port 7077
netstat -tuln | grep 7077
if [ $? -eq 0 ]; then
  echo "WARNING: Port 7077 is already in use!"
  echo "Attempting to kill any existing Spark master processes..."
  pkill -f "org.apache.spark.deploy.master.Master"
  sleep 5
fi

# Clear any previous work directories
echo "Cleaning up previous Spark work directories..."
rm -rf /tmp/spark-* 2>/dev/null

# Setup environment variables
export SPARK_LOCAL_IP=$POD_IP
export SPARK_MASTER_HOST=$POD_IP
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080

echo "Starting Spark Master with the following settings:"
echo "- SPARK_LOCAL_IP=$SPARK_LOCAL_IP"
echo "- SPARK_MASTER_HOST=$SPARK_MASTER_HOST"
echo "- SPARK_MASTER_PORT=$SPARK_MASTER_PORT"
echo "- SPARK_MASTER_WEBUI_PORT=$SPARK_MASTER_WEBUI_PORT"

# Try different methods to start Spark master
echo "Attempting to start Spark master..."

if [ -x "$SPARK_HOME/bin/spark-class" ]; then
  echo "Using $SPARK_HOME/bin/spark-class"
  $SPARK_HOME/bin/spark-class org.apache.spark.deploy.master.Master \
    --host $POD_IP \
    --port $SPARK_MASTER_PORT \
    --webui-port $SPARK_MASTER_WEBUI_PORT
elif [ -x "$SPARK_HOME/sbin/start-master.sh" ]; then
  echo "Using $SPARK_HOME/sbin/start-master.sh"
  # Modify the start script to use our IP
  sed -i "s/\${SPARK_MASTER_HOST:-\`hostname\`}/$POD_IP/g" $SPARK_HOME/sbin/start-master.sh 
  $SPARK_HOME/sbin/start-master.sh --host $POD_IP --port $SPARK_MASTER_PORT --webui-port $SPARK_MASTER_WEBUI_PORT
else
  echo "ERROR: Could not find spark-class or start-master.sh"
  echo "Checking Spark installation..."
  ls -la $SPARK_HOME/bin/
  ls -la $SPARK_HOME/sbin/
  exit 1
fi
