
import boto3
import os
import sys
from botocore.client import Config

# Configuration
MINIO_ENDPOINT = "http://localhost:9000"
ACCESS_KEY = "minio"
SECRET_KEY = "minio123"
BUCKET_NAME = "mlpipeline"

def fetch_results():
    if len(sys.argv) < 2:
        print("Usage: python fetch_results.py <pipeline_type>")
        return
        
    pipeline_type = sys.argv[1]
    last_run_file = f"metrics/last_run_id_{pipeline_type}.txt"
    
    if not os.path.exists(last_run_file):
        print(f"No run ID found for {pipeline_type}. Run automate_run.py first.")
        return
    
    with open(last_run_file, "r") as f:
        run_id = f.read().strip()
    
    print(f"Fetching results for Run ID: {run_id} ({pipeline_type})")
    
    s3 = boto3.resource('s3',
                        endpoint_url=MINIO_ENDPOINT,
                        aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY,
                        config=Config(signature_version='s3v4'),
                        region_name='us-east-1')
    
    bucket = s3.Bucket(BUCKET_NAME)
    
    # List objects with prefix
    found = False
    prefix = f"artifacts/{run_id}"
    
    print(f"Searching for artifacts in bucket '{BUCKET_NAME}'...")
    
    local_metrics_csv = f"metrics/metrics_{pipeline_type}_{run_id}.csv"
    if os.path.exists(local_metrics_csv):
        os.remove(local_metrics_csv)
        
    has_header = False
    
    for obj in bucket.objects.all():
        if run_id in obj.key:
            # Check for 'metrics' artifact (generic name in KFP v2) or 'metrics.csv'
            if obj.key.endswith("/metrics") or obj.key.endswith("metrics.csv") or obj.key.endswith("/metrics.csv"):
                print(f"Adding metrics from: {obj.key}")
                temp_file = "/tmp/temp_metrics.csv"
                bucket.download_file(obj.key, temp_file)
                
                with open(temp_file, 'r') as f_in:
                    lines = f_in.readlines()
                    if not lines:
                        continue
                        
                    with open(local_metrics_csv, 'a') as f_out:
                        # For the first file, keep the header. For others, skip it.
                        content_to_write = lines if not has_header else lines[1:]
                        f_out.writelines(content_to_write)
                        has_header = True
                
                os.remove(temp_file)
                found = True
            
            elif obj.key.endswith(".csv") and "metrics" not in obj.key:
                # Other CSV files (not the main metrics)
                local_name = os.path.basename(obj.key)
                print(f"Found artifact CSV: {local_name}")
                bucket.download_file(obj.key, local_name)
                found = True

    if found:
        # Sort the consolidated CSV by Round (index 0) and then by Mode (index 2)
        import csv
        with open(local_metrics_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        
        # Sort key: Round (as int), then Mode (EVAL before TRAIN or vice versa, TRAIN then EVAL is better)
        # mode order: TRAIN, then EVAL
        data.sort(key=lambda x: (int(x[0]), 0 if x[2] == 'TRAIN' else 1))
        
        with open(local_metrics_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

        print(f"✅ Consolidated and sorted results saved to {local_metrics_csv}")
    else:
        print("No metrics artifacts found for this run.")
        # Debug: list some files
        print("Sample files in bucket:")
        for i, obj in enumerate(bucket.objects.all()):
            if i < 10: print(obj.key)
            else: break

if __name__ == "__main__":
    fetch_results()
