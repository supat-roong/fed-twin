import sys
import os
from unittest.mock import patch, MagicMock, mock_open

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import fetch_results


@patch("boto3.resource")
@patch("os.path.exists")
@patch("os.remove")
@patch("sys.argv", ["fetch_results.py", "fl"])
def test_fetch_results_success(mock_remove, mock_exists, mock_boto3):
    # Setup mocks
    mock_exists.return_value = True

    # Mock boto3
    mock_s3 = mock_boto3.return_value
    mock_bucket = mock_s3.Bucket.return_value

    mock_obj1 = MagicMock()
    mock_obj1.key = "artifacts/run-123/metrics"
    mock_bucket.objects.all.return_value = [mock_obj1]

    csv_data = "Round,Episode,Mode,Reward,Loss\n1,0,TRAIN,5,1\n"

    # We need to mock open differently for different files
    # 1. Read last_run_id_fl.txt -> "run-123"
    # 2. Read temp_metrics.csv -> csv_data
    # 3. Read local_metrics_csv for sorting -> csv_data
    # 4. Write local_metrics_csv -> ...

    m = mock_open()
    m.side_effect = [
        mock_open(read_data="run-123\n").return_value,  # last_run_id
        mock_open(read_data=csv_data).return_value,  # temp_metrics.csv
        mock_open().return_value,  # f_out for append
        mock_open(read_data=csv_data).return_value,  # local_metrics_csv for sort read
        mock_open().return_value,  # local_metrics_csv for sort write
    ]

    with patch("builtins.open", m):
        fetch_results.fetch_results()

    # Verify
    mock_boto3.assert_called_once()
    mock_bucket.download_file.assert_called()


@patch("sys.argv", ["fetch_results.py"])
def test_fetch_results_no_args():
    fetch_results.fetch_results()


@patch("os.path.exists", return_value=False)
@patch("sys.argv", ["fetch_results.py", "fl"])
def test_fetch_results_no_last_run(mock_exists):
    fetch_results.fetch_results()


@patch("boto3.resource")
@patch("os.path.exists", return_value=True)
@patch("os.remove")
@patch("sys.argv", ["fetch_results.py", "fl"])
@patch("builtins.open", new_callable=mock_open, read_data="run-missing\n")
def test_fetch_results_no_artifacts_found(
    mock_file, mock_remove, mock_exists, mock_boto3
):
    mock_s3 = mock_boto3.return_value
    mock_bucket = mock_s3.Bucket.return_value
    mock_bucket.objects.all.return_value = []

    fetch_results.fetch_results()
    mock_boto3.assert_called_once()


@patch("boto3.resource")
@patch("os.path.exists", return_value=True)
@patch("os.remove")
@patch("sys.argv", ["fetch_results.py", "fl"])
def test_fetch_results_other_csv(mock_remove, mock_exists, mock_boto3):
    mock_s3 = mock_boto3.return_value
    mock_bucket = mock_s3.Bucket.return_value

    mock_obj1 = MagicMock()
    mock_obj1.key = "artifacts/run-123/other.csv"
    mock_bucket.objects.all.return_value = [mock_obj1]

    m = mock_open(read_data="run-123\n")
    with patch("builtins.open", m):
        fetch_results.fetch_results()

    mock_bucket.download_file.assert_called_with(
        "artifacts/run-123/other.csv", "other.csv"
    )
