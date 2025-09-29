"""Upload local data to GCS."""
import os
from pathlib import Path

from google.cloud import storage


def upload_directory_to_gcs(
    local_dir: str,
    bucket_name: str,
    gcs_prefix: str = "",
):
    """
    Upload all files in a directory to GCS.

    Args:
        local_dir: Local directory path
        bucket_name: GCS bucket name
        gcs_prefix: Prefix for GCS paths (e.g., "data/")
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    local_path = Path(local_dir)

    if not local_path.exists():
        print(f"Error: {local_dir} does not exist")
        return

    files = list(local_path.glob("*"))
    print(f"Found {len(files)} files in {local_dir}")

    for file_path in files:
        if file_path.is_file():
            blob_name = f"{gcs_prefix}{file_path.name}" if gcs_prefix else file_path.name
            blob = bucket.blob(blob_name)

            print(f"Uploading {file_path.name} to gs://{bucket_name}/{blob_name}...")
            blob.upload_from_filename(str(file_path))
            print(f"  ✓ Uploaded")

    print(f"\nAll files uploaded to gs://{bucket_name}/{gcs_prefix}")


def main():
    """Upload data to GCS."""
    # Get bucket name from env
    bucket_name = os.getenv("GCS_DATA_BUCKET")

    if not bucket_name:
        print("Error: GCS_DATA_BUCKET environment variable not set")
        print("Usage: export GCS_DATA_BUCKET=your-bucket-name")
        return

    # Upload data directory
    print(f"Uploading data to gs://{bucket_name}/...")
    upload_directory_to_gcs("data", bucket_name, gcs_prefix="")

    print("\nDone!")
    print(f"Data available at: gs://{bucket_name}/")


if __name__ == "__main__":
    main()

