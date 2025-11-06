#!/usr/bin/env python3
"""
Script to run pre-onboarding model by fetching JSON input from URL and saving results to database.

This script fetches JSON input from the IBV data API using IBVStatusID, runs the pre-onboarding
IA model, and saves results to SpeedyAnalysis table with model_type = 'PreOnboarding'.
"""

import asyncio
import gzip
import json
import logging
import os
import sys
import urllib
import warnings
from datetime import datetime

import dotenv
import pandas as pd
import requests
from sqlalchemy import create_engine, text

# Load environment variables from .env file
dotenv.load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pre-onboarding endpoint
PRE_ONBOARDING_ENDPOINT = "https://sim-pre-onboarding-model.thankfulforest-3cfea6f4.centralus.azurecontainerapps.io/model/v2/analyze"

# Health check endpoint for getting model version
HEALTH_CHECK_ENDPOINT = "https://sim-pre-onboarding-model.thankfulforest-3cfea6f4.centralus.azurecontainerapps.io/health_check"

# IBV data URL template
IBV_DATA_URL_TEMPLATE = (
    "https://lms.speedyloan.com/CommonServices.ashx?f=get_ibv_data&ibvsid={IBVStatusID}"
)


def get_model_version():
    """Get model version from the health_check API endpoint."""
    try:
        logger.debug("Fetching model version from health_check endpoint...")
        response = requests.get(HEALTH_CHECK_ENDPOINT, timeout=10)

        if response.status_code == 200:
            health_data = response.json()
            version = health_data.get("model_version", "unknown")
            logger.debug(f"Model version retrieved: {version}")
            return version
        else:
            logger.warning(f"Health check API returned status {response.status_code}")
            return "unknown"

    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not fetch model version from API: {e}")
        return "unknown"
    except Exception as e:
        logger.warning(f"Error getting model version: {e}")
        return "unknown"


def get_db_connection(database="BankuityPostOnboarding"):
    """Create database connection using environment variables."""
    try:
        server = "192.168.1.15"
        username = "azureuser"
        password = "$trongM0del413"
        odbc_driver = os.getenv("ODBC_DRIVER_VERSION", "ODBC Driver 18 for SQL Server")
        params = urllib.parse.quote_plus(
            f"DRIVER={{{odbc_driver}}};SERVER="
            + server
            + ";DATABASE="
            + database
            + ";UID="
            + username
            + ";PWD="
            + password
        )
        engine = create_engine(
            "mssql+pyodbc:///?odbc_connect=%s" % params,
            connect_args={"TrustServerCertificate": "yes"},
        )
        cnxn = engine.connect()
        return cnxn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def get_existing_ibv_tokens(conn, experiment_name):
    """Get list of IBVTokens that already exist in the database for the given experiment."""
    try:
        query = text("""
        SELECT DISTINCT IBVToken 
        FROM [dbo].[SpeedyAnalysis] 
        WHERE ExperimentName = :experiment_name
        AND ModelType = 'PreOnboarding'
        """)

        result = conn.execute(query, {"experiment_name": experiment_name})
        existing_tokens = {str(row[0]) for row in result.fetchall()}

        logger.info(
            f"Found {len(existing_tokens)} existing IBVTokens for pre-onboarding experiment '{experiment_name}'"
        )
        return existing_tokens

    except Exception as e:
        logger.error(f"Failed to query existing IBVTokens: {e}")
        return set()


def fetch_json_from_url(ibv_status_id):
    """
    Fetch JSON input data from the IBV data URL.

    Args:
        ibv_status_id: The IBVStatusID to fetch data for

    Returns:
        dict: JSON data from the API, or None if failed
    """
    try:
        url = IBV_DATA_URL_TEMPLATE.format(IBVStatusID=ibv_status_id)
        logger.info(f"Fetching JSON from URL: {url}")

        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            json_data = response.json()
            logger.info(
                f"Successfully fetched JSON data for IBVStatusID: {ibv_status_id}"
            )
            return json_data
        else:
            logger.error(
                f"Failed to fetch JSON. Status code: {response.status_code}, Response: {response.text}"
            )
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception while fetching JSON: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching JSON: {e}")
        return None


def fetch_model_request_from_db(ibv_status_id, date_created=None):
    """
    Fetch ModelRequest JSON from LMS_Logs..CModelLogs database.

    Args:
        ibv_status_id: The IBVStatusID to fetch data for
        date_created: Optional DateCreated to find the closest DateProcessed

    Returns:
        dict: JSON data from ModelRequest field, or None if failed
    """
    try:
        logger.info(
            f"Fetching ModelRequest from database for IBVStatusID: {ibv_status_id}"
        )

        # Connect to LMS_Logs database
        db_conn = get_db_connection(database="LMS_Logs")

        try:
            query = text("""
            SELECT ModelName, DateProcessed, ModelRequest 
            FROM CModelLogs 
            WHERE IBVStatusID = :ibv_status_id
            ORDER BY DateProcessed DESC
            """)

            result = db_conn.execute(query, {"ibv_status_id": ibv_status_id})
            rows = result.fetchall()

            if not rows:
                logger.error(f"No ModelRequest found for IBVStatusID: {ibv_status_id}")
                return None

            # If date_created is provided, find the row with DateProcessed closest to it
            if date_created and len(rows) > 1:
                try:
                    target_date = pd.to_datetime(date_created)
                    closest_row = None
                    min_diff = None

                    for row in rows:
                        date_processed = pd.to_datetime(row[1])
                        diff = abs((date_processed - target_date).total_seconds())

                        if min_diff is None or diff < min_diff:
                            min_diff = diff
                            closest_row = row

                    selected_row = closest_row
                    logger.info(
                        f"Selected row with DateProcessed closest to DateCreated: {selected_row[1]}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error finding closest date, using most recent: {e}"
                    )
                    selected_row = rows[0]
            else:
                # Use the most recent row (already sorted by DateProcessed DESC)
                selected_row = rows[0]
                if len(rows) > 1:
                    logger.info(
                        f"Multiple rows found, using most recent DateProcessed: {selected_row[1]}"
                    )

            model_request = selected_row[2]

            if model_request is None:
                logger.error(f"ModelRequest is NULL for IBVStatusID: {ibv_status_id}")
                return None

            # Parse the ModelRequest JSON
            if isinstance(model_request, str):
                json_data = json.loads(model_request)
            else:
                # If it's already a dict or bytes, handle accordingly
                json_data = (
                    json.loads(model_request.decode("utf-8"))
                    if isinstance(model_request, bytes)
                    else model_request
                )

            logger.info(
                f"Successfully fetched ModelRequest for IBVStatusID: {ibv_status_id}"
            )
            return json_data

        finally:
            db_conn.close()

    except Exception as e:
        logger.error(f"Error fetching ModelRequest from database: {e}")
        return None


def run_model_api(json_input):
    """
    Run the pre-onboarding model API with the given JSON input.

    Args:
        json_input: The JSON input data for the model

    Returns:
        dict: Model result, or None if failed
    """
    try:
        headers = {"Content-Type": "application/json"}

        payload = {
            "input": json.dumps(json_input),
        }

        logger.info("Calling pre-onboarding model API...")
        response = requests.post(
            url=PRE_ONBOARDING_ENDPOINT,
            data=json.dumps(payload),
            headers=headers,
            timeout=1200,
        )

        if response.status_code == 200:
            result = response.json()
            logger.info("Pre-onboarding model completed successfully")
            return result
        else:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling model API: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling model API: {e}")
        return None


def insert_result_to_db(
    conn,
    result_data,
    request_data,
    ibv_status_id,
    experiment_name,
    client_name,
    ibv_name,
    model_version,
):
    """Insert result into SpeedyAnalysis table."""
    try:
        # Convert result and request to JSON strings
        json_response = json.dumps(result_data, default=str)
        json_request = json.dumps(request_data, default=str)

        # Compress the response
        compressed_response = gzip.compress(json_response.encode("utf-8"))
        compressed_request = gzip.compress(json_request.encode("utf-8"))

        insert_query = text("""
        INSERT INTO [dbo].[SpeedyAnalysis] 
        (ClientName, ModelVersion, ExperimentName, IBVName, IBVToken, ModelType, RequestJSON, ResponseJSON)
        VALUES (:client_name, :model_version, :experiment_name, :ibv_name, :ibv_token, :model_type, :request_json, :response_json)
        """)

        conn.execute(
            insert_query,
            {
                "client_name": client_name,
                "model_version": model_version,
                "experiment_name": experiment_name,
                "ibv_name": ibv_name,
                "ibv_token": str(ibv_status_id),
                "model_type": "PreOnboarding",
                "request_json": compressed_request,
                "response_json": compressed_response,
            },
        )

        conn.commit()
        logger.info(
            f"Successfully saved result to database for IBVStatusID: {ibv_status_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to insert result to database: {e}")
        conn.rollback()
        return False


def process_ibv_status_id(
    ibv_status_id, experiment_name, client_name, ibv_name, asOfDate=None
):
    """
    Process a single IBVStatusID by fetching JSON, running model, and saving to DB.

    Args:
        ibv_status_id: The IBVStatusID to process
        experiment_name: Name of the experiment
        client_name: Name of the client
        ibv_name: Name of the IBV
        asOfDate: The asOfDate to use in the JSON input (date string in YYYY-MM-DD format)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("=" * 60)
        logger.info(f"Processing IBVStatusID: {ibv_status_id}")
        logger.info("=" * 60)

        # Step 1: Fetch JSON from database (CModelLogs)
        json_input = fetch_model_request_from_db(ibv_status_id, asOfDate)
        if json_input is None:
            logger.error("Failed to fetch ModelRequest from database")
            return False

        # Step 1.5: Validate JSON has required key
        if "applicationInformation" not in json_input:
            logger.warning(
                f"Skipping IBVStatusID {ibv_status_id}: JSON does not contain 'applicationInformation' key"
            )
            return False

        # Step 1.6: Override asOfDate if provided (COMMENTED OUT)
        # if asOfDate is not None:
        #     logger.info(f"Overwriting asOfDate with: {asOfDate}")
        #     json_input["asOfDate"] = asOfDate

        # Step 2: Run the model
        result_data = run_model_api(json_input)
        if result_data is None:
            logger.error("Failed to run model")
            return False

        # Add metadata to result
        result_data["processing_timestamp"] = datetime.now().isoformat()
        result_data["experiment_name"] = experiment_name
        result_data["client_name"] = client_name
        result_data["ibv_name"] = ibv_name
        result_data["ibv_status_id"] = ibv_status_id

        # Step 3: Save to database
        model_version = get_model_version()
        db_conn = get_db_connection()

        try:
            success = insert_result_to_db(
                db_conn,
                result_data,
                json_input,
                ibv_status_id,
                experiment_name,
                client_name,
                ibv_name,
                model_version,
            )

            if success:
                logger.info(f"Successfully processed IBVStatusID: {ibv_status_id}")
                return True
            else:
                logger.error(
                    f"Failed to save to database for IBVStatusID: {ibv_status_id}"
                )
                return False

        finally:
            db_conn.close()

    except Exception as e:
        logger.error(
            f"Exception occurred while processing IBVStatusID {ibv_status_id}: {str(e)}"
        )
        return False


async def process_ibv_status_id_async(
    index, ibv_status_id, experiment_name, client_name, ibv_name, asOfDate, semaphore
):
    """
    Async wrapper that offloads the blocking work into a thread, respecting the semaphore.

    Args:
        index: Index of the current item being processed
        ibv_status_id: The IBVStatusID to process
        experiment_name: Name of the experiment
        client_name: Name of the client
        ibv_name: Name of the IBV
        asOfDate: The asOfDate to use (DateCreated)
        semaphore: Asyncio semaphore for concurrency control

    Returns:
        tuple: (index, ibv_status_id, success)
    """
    async with semaphore:
        success = await asyncio.to_thread(
            process_ibv_status_id,
            ibv_status_id,
            experiment_name,
            client_name,
            ibv_name,
            asOfDate,
        )
        return index, ibv_status_id, success


def process_csv_file(csv_path, experiment_name, client_name, ibv_name, test_mode=False):
    """
    Process all rows in a CSV file.

    Args:
        csv_path: Path to the CSV file containing IBVStatusID column
        experiment_name: Name of the experiment
        client_name: Name of the client
        ibv_name: Name of the IBV
        test_mode: If True, only process first 10 rows

    Returns:
        dict: Summary of processing results
    """
    try:
        # Read the CSV file
        logger.info("=" * 80)
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate that IBVStatusID column exists
        if "IBVStatusID" not in df.columns:
            logger.error("CSV file must contain 'IBVStatusID' column")
            return {
                "total_rows": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "already_processed": 0,
                "error": "Missing IBVStatusID column",
            }

        total_rows = len(df)
        logger.info(f"Total rows in CSV: {total_rows}")

        # Get existing IBVTokens to avoid reprocessing
        logger.info("Checking for already processed IBVStatusIDs...")
        db_conn = get_db_connection()
        try:
            existing_ibv_tokens = get_existing_ibv_tokens(db_conn, experiment_name)
        finally:
            db_conn.close()

        # Filter out rows that have already been processed

        # Apply test mode filter
        if test_mode:
            df = df.head(min(10, len(df)))
            logger.info("Test mode enabled - processing only first 10 rows")

        initial_count = len(df)
        df = df[~df["IBVStatusID"].astype(str).isin(existing_ibv_tokens)]
        already_processed = initial_count - len(df)

        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Remaining to process: {len(df)}")

        if len(df) == 0:
            logger.warning(
                "All rows have already been processed for this experiment. Exiting."
            )
            return {
                "total_rows": total_rows,
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "already_processed": already_processed,
            }

        logger.info(f"Processing {len(df)} rows")
        logger.info("=" * 80)

        # Track results
        successful = 0
        failed = 0
        skipped = 0
        total_to_process = len(df)
        current_file = 0

        # Process each row
        for index, row in df.iterrows():
            try:
                current_file += 1
                ibv_status_id = row["IBVStatusID"]

                # Skip if IBVStatusID is null or empty
                if pd.isna(ibv_status_id):
                    logger.warning(
                        f"[{current_file}/{total_to_process}] Row {index + 1}: Skipping - IBVStatusID is null"
                    )
                    skipped += 1
                    continue

                # Convert to int
                ibv_status_id = int(ibv_status_id)

                # Extract and parse DateCreated as date (not datetime)
                asOfDate = None
                if "DateCreated" in df.columns and not pd.isna(row["DateCreated"]):
                    try:
                        # Parse DateCreated and convert to date string (YYYY-MM-DD)
                        date_created = pd.to_datetime(row["DateCreated"])
                        asOfDate = date_created.strftime("%Y-%m-%d")
                        logger.info(f"Using DateCreated as asOfDate: {asOfDate}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse DateCreated: {e}, will use default asOfDate"
                        )
                        asOfDate = None

                logger.info(f"\n{'='*80}")
                logger.info(
                    f"[{current_file}/{total_to_process}] Processing IBVStatusID {ibv_status_id} (CSV row {index + 1})"
                )
                logger.info(f"{'='*80}")

                # Process the IBVStatusID
                success = process_ibv_status_id(
                    ibv_status_id, experiment_name, client_name, ibv_name, asOfDate
                )

                if success:
                    successful += 1
                    logger.info(
                        f"✓ [{current_file}/{total_to_process}] Completed successfully | Success: {successful}, Failed: {failed}, Skipped: {skipped}"
                    )
                else:
                    failed += 1
                    logger.error(
                        f"✗ [{current_file}/{total_to_process}] Failed | Success: {successful}, Failed: {failed}, Skipped: {skipped}"
                    )

            except Exception as e:
                logger.error(
                    f"[{current_file}/{total_to_process}] Error processing row {index + 1}: {str(e)}"
                )
                failed += 1

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rows in CSV: {total_rows}")
        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Rows processed in this run: {len(df)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info("=" * 80)

        return {
            "total_rows": total_rows,
            "already_processed": already_processed,
            "processed": len(df),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
        }

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return {
            "total_rows": 0,
            "already_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "error": "File not found",
        }
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return {
            "total_rows": 0,
            "already_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "error": str(e),
        }


async def process_csv_file_parallel(
    csv_path,
    experiment_name,
    client_name,
    ibv_name,
    test_mode=False,
    concurrency_limit=50,
):
    """
    Process all rows in a CSV file using parallel/async processing.

    Args:
        csv_path: Path to the CSV file containing IBVStatusID column
        experiment_name: Name of the experiment
        client_name: Name of the client
        ibv_name: Name of the IBV
        test_mode: If True, only process first 10 rows
        concurrency_limit: Maximum number of concurrent requests

    Returns:
        dict: Summary of processing results
    """
    try:
        # Read the CSV file
        logger.info("=" * 80)
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate that IBVStatusID column exists
        if "IBVStatusID" not in df.columns:
            logger.error("CSV file must contain 'IBVStatusID' column")
            return {
                "total_rows": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "already_processed": 0,
                "error": "Missing IBVStatusID column",
            }

        total_rows = len(df)
        logger.info(f"Total rows in CSV: {total_rows}")

        # Get existing IBVTokens to avoid reprocessing
        logger.info("Checking for already processed IBVStatusIDs...")
        db_conn = get_db_connection()
        try:
            existing_ibv_tokens = get_existing_ibv_tokens(db_conn, experiment_name)
        finally:
            db_conn.close()

        # Apply test mode filter
        if test_mode:
            df = df.head(min(10, len(df)))
            logger.info("Test mode enabled - processing only first 10 rows")

        initial_count = len(df)
        df = df[~df["IBVStatusID"].astype(str).isin(existing_ibv_tokens)]
        already_processed = initial_count - len(df)

        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Remaining to process: {len(df)}")

        if len(df) == 0:
            logger.warning(
                "All rows have already been processed for this experiment. Exiting."
            )
            return {
                "total_rows": total_rows,
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "already_processed": already_processed,
            }

        logger.info(
            f"Processing {len(df)} rows with concurrency limit: {concurrency_limit}"
        )
        logger.info("=" * 80)

        # Prepare tasks
        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit)

        for index, row in df.iterrows():
            try:
                ibv_status_id = row["IBVStatusID"]

                # Skip if IBVStatusID is null or empty
                if pd.isna(ibv_status_id):
                    logger.warning(f"Row {index + 1}: Skipping - IBVStatusID is null")
                    continue

                # Convert to int
                ibv_status_id = int(ibv_status_id)

                # Extract and parse DateCreated as date (not datetime)
                asOfDate = None
                if "DateCreated" in df.columns and not pd.isna(row["DateCreated"]):
                    try:
                        # Parse DateCreated and convert to date string (YYYY-MM-DD)
                        date_created = pd.to_datetime(row["DateCreated"])
                        asOfDate = date_created.strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse DateCreated: {e}, will use default asOfDate"
                        )
                        asOfDate = None

                # Create async task
                task = asyncio.create_task(
                    process_ibv_status_id_async(
                        index,
                        ibv_status_id,
                        experiment_name,
                        client_name,
                        ibv_name,
                        asOfDate,
                        semaphore,
                    )
                )
                tasks.append(task)

            except Exception as e:
                logger.error(f"Error preparing row {index + 1}: {str(e)}")

        # Execute all tasks in parallel
        logger.info(f"Starting parallel processing of {len(tasks)} tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Track results
        successful = 0
        failed = 0
        skipped = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed += 1
            else:
                index, ibv_status_id, success = result
                if success:
                    successful += 1
                    logger.info(f"✓ IBVStatusID {ibv_status_id} completed successfully")
                else:
                    failed += 1
                    logger.error(f"✗ IBVStatusID {ibv_status_id} failed")

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rows in CSV: {total_rows}")
        logger.info(f"Already processed: {already_processed}")
        logger.info(f"Rows processed in this run: {len(tasks)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Skipped: {skipped}")
        logger.info("=" * 80)

        return {
            "total_rows": total_rows,
            "already_processed": already_processed,
            "processed": len(tasks),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
        }

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        return {
            "total_rows": 0,
            "already_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "error": "File not found",
        }
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return {
            "total_rows": 0,
            "already_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "error": str(e),
        }


def main():
    """
    Main function to run the script.

    Usage:
        Set environment variables in .env file:
        - DATAPATH: Path to the CSV file containing IBVStatusID column
        - EXPERIMENT_NAME: Name of the experiment
        - CLIENT_NAME: Name of the client
        - IBV_NAME: Name of the IBV
        - TEST: If "true", only process first 10 rows (default: false)
        - PARALLEL: If "true", use parallel processing (default: false)
        - CONCURRENCY_LIMIT: Number of concurrent requests when using parallel processing (default: 50)

        Or pass CSV path as command line argument:
        python rerun_model.py <csv_path>
    """
    # Get configuration from environment variables
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.getenv("DATAPATH")

    if not csv_path:
        logger.error(
            "CSV file path not provided. Set DATAPATH environment variable or pass as command line argument."
        )
        sys.exit(1)

    # Get other configuration from environment variables
    experiment_name = os.getenv("EXPERIMENT_NAME")
    client_name = os.getenv("CLIENT_NAME")
    ibv_name = os.getenv("IBV_NAME")
    test_mode = os.getenv("TEST", "false").lower() == "true"
    parallel_mode = os.getenv("PARALLEL", "false").lower() == "true"
    concurrency_limit = int(os.getenv("CONCURRENCY_LIMIT", "50"))

    # Validate required environment variables
    if not experiment_name or not client_name or not ibv_name:
        logger.error(
            "Missing required environment variables: EXPERIMENT_NAME, CLIENT_NAME, IBV_NAME"
        )
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"CSV File Path: {csv_path}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Client Name: {client_name}")
    logger.info(f"IBV Name: {ibv_name}")
    logger.info(f"Test Mode: {test_mode}")
    logger.info(f"Parallel Mode: {parallel_mode}")
    if parallel_mode:
        logger.info(f"Concurrency Limit: {concurrency_limit}")
    logger.info("=" * 80)

    # Process the CSV file
    if parallel_mode:
        # Use async parallel processing
        summary = asyncio.run(
            process_csv_file_parallel(
                csv_path,
                experiment_name,
                client_name,
                ibv_name,
                test_mode,
                concurrency_limit,
            )
        )
    else:
        # Use sequential processing
        summary = process_csv_file(
            csv_path, experiment_name, client_name, ibv_name, test_mode
        )

    # Exit based on results
    if "error" in summary:
        logger.error(f"Script failed: {summary['error']}")
        sys.exit(1)
    elif summary["failed"] > 0:
        logger.warning(
            f"Script completed with {summary['failed']} failure(s) out of {summary['processed']} processed rows"
        )
        sys.exit(1)
    else:
        logger.info(
            f"Script completed successfully - {summary['successful']} rows processed"
        )
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        sys.exit(1)
