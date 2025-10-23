import boto3
from botocore.exceptions import NoCredentialsError
import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_and_upload_plot(
    plt_obj,
    bucket_name,
    folder="ml_outputs",
    filename="confusion_matrix.png",
    expiration=3600
):
    """
    Saves a matplotlib plot locally, uploads it to S3, and returns a presigned URL.
    
    Args:
        plt_obj: The matplotlib.pyplot module or figure object.
        bucket_name: Your S3 bucket name.
        folder: Folder path in S3 (default: 'ml_outputs').
        filename: Name for the local and S3 file.
        expiration: Presigned URL expiry in seconds (default: 1 hour).

    Returns:
        str: Presigned URL for the uploaded file.
    """
    # Timestamped file name to avoid overwrites
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(filename)
    filename = f"{base}_{timestamp}{ext}"

    # Save locally
    plt_obj.savefig(filename, bbox_inches="tight")
    plt_obj.close()
    print(f"Plot saved locally as {filename}")

    # Upload to S3
    s3 = boto3.client("s3")
    object_name = f"{folder}/{filename}"

    try:
        s3.upload_file(filename, bucket_name, object_name)
        print(f"Uploaded to s3://{bucket_name}/{object_name}")

        # Generate presigned URL
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_name},
            ExpiresIn=expiration
        )
        print(f"Presigned URL (valid {expiration/60:.0f} min): {url}")
        return url

    except FileNotFoundError:
        print("The file was not found.")
    except NoCredentialsError:
        print("AWS credentials not available.")
    
    return None
