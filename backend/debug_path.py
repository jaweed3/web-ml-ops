import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Pastikan environment variable termuat
load_dotenv() 

def inspect_model_source():
    # Setup koneksi
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("CRITICAL: MLFLOW_TRACKING_URI not set in .env")
        return

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    model_name = "ElasticNetModel" # Sesuaikan jika beda
    
    print(f"--- INSPECTING REGISTRY: {model_name} ---")
    
    try:
        # Ambil versi terakhir
        latest_versions = client.get_latest_versions(model_name)
        if not latest_versions:
            print("No versions found.")
            return

        # Kita cek 3 versi terakhir untuk melihat pola
        for v in latest_versions[-3:]: 
            print(f"\nVersion: {v.version}")
            print(f"Stage: {v.current_stage}")
            print(f"Status: {v.status}")
            print(f"Source (Artifact URI): {v.source}") # <--- INI TERSANGKANYA
            
            if v.source.startswith("dbfs:") or "dagshub" in v.source or "s3" in v.source:
                print("   -> Tipe: Remote Cloud (Seharusnya AMAN)")
            elif v.source.startswith("file:"):
                print("   -> Tipe: Local Filesystem (BAHAYA: Tidak bisa di-load dari mesin lain)")
            elif v.source.startswith("models:"):
                 print("   -> Tipe: Virtual Reference (Indikasi Log Error/Circular)")
            else:
                print("   -> Tipe: Unknown")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_model_source()
