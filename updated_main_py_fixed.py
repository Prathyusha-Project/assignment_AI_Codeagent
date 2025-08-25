# app/main.py

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from config import IS_READY, start_time, log
import uvicorn
import os
import json
import pandas as pd
import re
import asyncio
import uuid
import traceback
import base64
import urllib.parse
import aiohttp
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from typing import List
import time
from dotenv import load_dotenv
import gc  # For garbage collection

load_dotenv()

from redis.asyncio import Redis
from middleware.redis_session_middleware import RedisSessionMiddleware
from middleware.auth_required import AuthRequiredMiddleware

# Import database and storage
from cosmosdb import container, metadata_fields_container
from blobstorage import blob_container_client
from azure.cosmos import exceptions

# Import route modules
from routes.health import router as health_router
from routes.auth import router as auth_router
from routes.groups import router as groups_router
from routes.bots import router as bots_router
from routes.file import router as file_router
from routes.status import router as status_router
from routes.chat import router as chat_router
from routes.brd import router as brd_router
from routes.metadata import router as metadata_router
from routes.user_files import router as user_files_router

# Import shared utilities from metadata module
from routes.metadata import (
    generate_feedback_summary, 
    MetadataExtractionResult, 
    MetadataFieldResult, 
    extract_metadata_for_text_with_retry, 
    MetadataField, 
    ExampleRow, 
    MAX_ROWS_LIMIT,
    determine_instance_strategy, 
    setup_client_pool_for_strategy
)

# Shared constants
BLOB_EXPIRY_DAYS = 90

app = FastAPI(
    title="ENB Copilot Web API",
    description="A Python API to serve as Backend For the ENB Copilot Web App",
    version="0.1.0",
    docs_url="/docs",
)

# --- CORS Middleware ---
origins = [
    "http://localhost:5173",  # Frontend origin
]

# 1) Starlette's built-in SessionMiddleware (cookie-based fallback)
app.add_middleware(
    SessionMiddleware,
    secret_key="4224dd52-f701-47a4-9d9e-f7c4e1ca92b9",
    session_cookie="enb_session",
    max_age=86400,
)

# 2) Redis-backed Session Middleware
redis_client = Redis.from_url(os.getenv("REDIS_URL"), encoding="utf-8", decode_responses=True)

app.add_middleware(
    RedisSessionMiddleware,
    redis=redis_client,
)

# 3) CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include Routers ---
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(groups_router)
app.include_router(bots_router)
app.include_router(file_router)
app.include_router(status_router)
app.include_router(chat_router)
app.include_router(brd_router)
app.include_router(metadata_router)
app.include_router(user_files_router)

# --- Root Redirect & Static Files ---
from fastapi.responses import RedirectResponse

# Utility functions
def get_user_email(request: Request) -> str:
    """Extract user email from request session - centralized logic"""
    # Get user email from session/request
    user = getattr(request.state, "user", None)
    user_email = user.email if user and hasattr(user, "email") else None

    if not user_email:
        session = getattr(request.state, "session", {})
        user_profile = session.get("user")
        if user_profile:
            user_email = user_profile.get("mail") or user_profile.get("userPrincipalName")

    if not user_email:
        # Try getting from account info
        session = getattr(request.state, "session", {})
        account = session.get("account") if session else None
        if account:
            user_email = account.get("preferred_username") or account.get("email") or account.get("upn")

    return user_email

def normalize_field_name(name: str) -> str:
    """Normalize field names to handle common misspellings and formatting issues"""
    # Basic normalization - trim and replace multiple spaces with single space
    name = name.strip().replace("  ", " ")
    # Common misspellings dictionary - only apply to whole words to avoid corruption
    common_misspellings = {
        'amnufacturer': 'manufacturer',
        'manufacutrer': 'manufacturer',
        'manufactuerer': 'manufacturer',
        'mannufacturer': 'manufacturer',
        'qty': 'quantity',
        'quantit': 'quantity',
        'nmber': 'number',
        'numbr': 'number',
        'seral': 'serial',
        'seriel': 'serial'
    }

    # Split into words and process each word individually to avoid corruption
    words = name.split()
    processed_words = []
    for word in words:
        word_lower = word.lower()
        # Only replace if the entire word matches a misspelling
        if word_lower in common_misspellings:
            processed_words.append(common_misspellings[word_lower])
        else:
            processed_words.append(word)

    return ' '.join(processed_words)

class MockUploadFile:
    """Mock UploadFile class for loaded files from blob storage"""
    def __init__(self, content: bytes, filename: str):
        self.content = content
        self.filename = filename
        self.size = len(content)
        self.content_type = "application/octet-stream"
        self._position = 0

    async def read(self) -> bytes:
        return self.content

    async def seek(self, position: int) -> None:
        """Reset file pointer - needed for file saving operations"""
        self._position = position
        return None

@app.get("/api/history")
async def get_run_history(request: Request):
    """Get all run history for a user"""
    try:
        user_email = get_user_email(request)
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        print(f"Fetching history for user: {user_email}")

        # Query Cosmos DB for all user's records
        query = f"SELECT * FROM c WHERE c.userEmail = '{user_email}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        # Transform items to include only necessary information and distinguish drafts from runs
        history_items = []
        for item in items:
            # Determine if this is a draft or completed run
            is_draft = item.get("isDraft", False) or item.get("status") == "draft"
            is_completed = item.get("isCompleted", False) or item.get("status") == "completed"

            history_item = {
                "id": item["id"],
                "timestamp": item["timestamp"],
                "reportName": item.get("extractionReportName", "Untitled Report"),
                "uploadedFiles": [file["fileName"] for file in item.get("uploadedFiles", [])],
                "extractedFile": item.get("extractedFile", {}).get("fileName") if "extractedFile" in item else None,
                "numResults": len(item.get("extractedData", {}).get("tableData", [])) if "extractedData" in item else 0,
                "isDraft": is_draft,
                "isCompleted": is_completed,
                "status": item.get("status", "completed" if is_completed else ("draft" if is_draft else "unknown"))
            }
            history_items.append(history_item)

        print(f"Found {len(history_items)} total items")

        # Sort by timestamp descending (newest first)
        history_items.sort(key=lambda x: x["timestamp"], reverse=True)

        return {
            "status": "success",
            "items": history_items
        }

    except Exception as e:
        print(f"Error getting run history: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get run history: {str(e)}")

@app.delete("/api/history")
@app.delete("/api/metadata/history")
async def clear_run_history(request: Request):
    """Delete all run history for a user including files and metadata"""
    try:
        # Get request body for userEmail if provided
        try:
            body = await request.json()
            user_email_from_body = body.get("userEmail") if body else None
        except:
            user_email_from_body = None

        user_email = get_user_email(request) or user_email_from_body

        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        print(f"Starting deletion for user: {user_email}")

        # Query Cosmos DB for user's RUN HISTORY records only (NOT metadata configurations)
        query = f"SELECT * FROM c WHERE c.userEmail = '{user_email}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        print(f"Found {len(items)} run history records in Cosmos DB for user {user_email}")
        print("NOTE: Metadata field configurations are preserved and NOT deleted")

        # Delete blobs from Azure Storage
        print(f"Starting blob deletion process...")
        blob_paths_to_delete = set()

        for item in items:
            # Get uploaded file paths
            if "uploadedFiles" in item:
                for file_info in item["uploadedFiles"]:
                    if "blobPath" in file_info:
                        blob_paths_to_delete.add(file_info["blobPath"])

            # Get extracted CSV file path
            if "extractedFile" in item and "blobPath" in item["extractedFile"]:
                blob_paths_to_delete.add(item["extractedFile"]["blobPath"])

        print(f"Found {len(blob_paths_to_delete)} blobs to delete")

        # Delete blobs with retry logic
        deleted_blobs = 0
        for blob_path in blob_paths_to_delete:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    blob_client = blob_container_client.get_blob_client(blob_path)
                    if blob_client.exists():
                        blob_client.delete_blob(delete_snapshots="include")
                        deleted_blobs += 1
                        print(f"Successfully deleted blob: {blob_path}")
                        break
                    else:
                        print(f"Blob not found: {blob_path}")
                        break
                except Exception as e:
                    retry_count += 1
                    print(f"Attempt {retry_count} - Failed to delete blob {blob_path}: {str(e)}")
                    if retry_count < max_retries:
                        time.sleep(1)
                    else:
                        print(f"Failed to delete blob after {max_retries} attempts: {blob_path}")

        # Delete Cosmos DB run history records only
        deleted_records = 0
        for item in items:
            try:
                container.delete_item(item=item["id"], partition_key=user_email)
                deleted_records += 1
                print(f"Deleted run history record: {item['id']}")
            except Exception as e:
                print(f"Warning: Failed to delete Cosmos DB item {item['id']}: {str(e)}")
                if "Not Found" not in str(e):
                    raise e

        return {
            "status": "success",
            "message": f"Successfully deleted {deleted_records} run history records and {deleted_blobs} associated files. Metadata field configurations preserved.",
            "deletedRecords": deleted_records,
            "deletedFiles": deleted_blobs,
            "note": "User's saved metadata configurations were NOT deleted and remain available for future use."
        }

    except Exception as e:
        print(f"Error clearing run history: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to clear run history: {str(e)}")

@app.delete("/api/history/{run_id}")
async def delete_run_history_item(run_id: str, request: Request):
    """Delete a specific run history item for a user"""
    try:
        user_email = request.query_params.get("userEmail") or get_user_email(request)

        if not user_email:
            raise HTTPException(status_code=400, detail="User email is required")

        print(f"Attempting to delete run history item {run_id} for user: {user_email}")

        try:
            # First check if the item exists and belongs to the user
            existing_item = container.read_item(item=run_id, partition_key=user_email)
            print(f"Found item to delete: {existing_item.get('reportName', 'Unknown')}")

            # Delete the item
            container.delete_item(item=run_id, partition_key=user_email)
            print(f"Successfully deleted run history item: {run_id}")

            return {
                "message": f"Run history item deleted successfully",
                "deleted_item_id": run_id,
                "user_email": user_email
            }

        except Exception as cosmos_error:
            if "NotFound" in str(cosmos_error):
                raise HTTPException(status_code=404, detail=f"Run history item not found or does not belong to user")
            else:
                raise cosmos_error

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error deleting run history item: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete run history item: {str(e)}")

@app.post("/api/metadata")
async def save_metadata_fields(request: Request):
    """Save metadata fields configuration for a user"""
    try:
        body = await request.json()
        user_email = get_user_email(request) or body.get("userEmail")

        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        print(f"Using user email for saving metadata: {user_email}")

        # Handle the frontend format: array of configurations in metadata field
        if "metadata" in body:
            metadata_content = body["metadata"]
            if isinstance(metadata_content, list):
                print(f"Processing {len(metadata_content)} configurations from frontend")

                # Clear existing configurations for this user
                try:
                    query = f"SELECT c.id FROM c WHERE c.userEmail = '{user_email}'"
                    existing_items = list(metadata_fields_container.query_items(query=query, enable_cross_partition_query=True))
                    for item in existing_items:
                        metadata_fields_container.delete_item(item=item["id"], partition_key=user_email)
                    print(f"Cleared {len(existing_items)} existing configurations")
                except Exception as e:
                    print(f"Warning: Failed to clear existing configurations: {str(e)}")

                # Save each configuration in the array
                saved_configs = []
                for config in metadata_content:
                    if isinstance(config, dict) and "name" in config and "fieldsData" in config:
                        config_name = config["name"]
                        fields_data = config["fieldsData"]

                        doc_id = str(uuid.uuid4())
                        doc = {
                            "id": doc_id,
                            "userEmail": user_email,
                            "timestamp": datetime.utcnow().isoformat(),
                            "configurationName": config_name,
                            "freeTextDescription": fields_data.get("freeTextDesc", ""),
                            "metadataFields": fields_data.get("metaDataFields", []),
                            "expectingTableOutput": {"tableData": True}
                        }

                        metadata_fields_container.create_item(body=doc)
                        saved_configs.append({"id": doc_id, "name": config_name})
                        print(f"Saved configuration: {config_name} with ID: {doc_id}")

                return {
                    "status": "success",
                    "message": f"Successfully saved {len(saved_configs)} metadata configurations",
                    "configurations": saved_configs
                }

        # Handle single configuration (legacy support)
        metadata_content = body.get("metadata", body)
        freeTextDescription = metadata_content.get("freeTextDescription", "")
        metadataFields = metadata_content.get("metadataFields", [])
        expectingTableOutput = metadata_content.get("expectingTableOutput", {"tableData": True})
        configurationName = metadata_content.get("configurationName", "Default Configuration")

        doc_id = str(uuid.uuid4())
        doc = {
            "id": doc_id,
            "userEmail": user_email,
            "timestamp": datetime.utcnow().isoformat(),
            "freeTextDescription": freeTextDescription,
            "metadataFields": metadataFields,
            "expectingTableOutput": expectingTableOutput,
            "configurationName": configurationName
        }

        metadata_fields_container.create_item(body=doc)

        return {
            "status": "success",
            "message": "Metadata fields configuration saved successfully",
            "id": doc_id
        }

    except Exception as e:
        print(f"Error saving metadata fields: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to save metadata fields: {str(e)}")

@app.get("/api/metadata")
async def get_metadata_fields(request: Request):
    """Get all saved metadata fields configurations for a user"""
    try:
        user_email = get_user_email(request) or request.query_params.get("userEmail")

        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        print(f"Fetching metadata fields for user: {user_email}")

        # Query Cosmos DB for all user's metadata field configurations
        query = f"SELECT * FROM c WHERE c.userEmail = '{user_email}'"
        items = list(metadata_fields_container.query_items(query=query, enable_cross_partition_query=True))

        # Transform items to match the frontend expected format
        configurations = []
        for item in items:
            # Safely get the first metadata field name
            first_field_name = "Long Description"
            if item.get("metadataFields") and len(item["metadataFields"]) > 0:
                first_field = item["metadataFields"][0]
                if isinstance(first_field, dict) and "name" in first_field:
                    first_field_name = first_field["name"]

            config = {
                "name": item.get("configurationName", "Default Configuration"),
                "id": item["id"],
                "fieldsData": {
                    "freeTextDesc": item.get("freeTextDescription", ""),
                    "freeTextColName": first_field_name,
                    "metaDataFields": item.get("metadataFields", []),
                    "lastUpdated": item["timestamp"],
                    "userEmail": user_email
                }
            }
            configurations.append(config)

        # Sort by timestamp descending (newest first)
        configurations.sort(key=lambda x: x["fieldsData"]["lastUpdated"], reverse=True)
        return configurations

    except Exception as e:
        print(f"Error getting metadata fields: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata fields: {str(e)}")

@app.get("/api/metadata/{config_id}")
async def get_metadata_fields_by_id(config_id: str, request: Request):
    """Get a specific metadata fields configuration by ID"""
    try:
        user_email = get_user_email(request)
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        try:
            item = metadata_fields_container.read_item(item=config_id, partition_key=user_email)
            config = {
                "id": item["id"],
                "timestamp": item["timestamp"],
                "configurationName": item.get("configurationName", "Default Configuration"),
                "freeTextDescription": item["freeTextDescription"],
                "metadataFields": item["metadataFields"],
                "expectingTableOutput": item["expectingTableOutput"]
            }

            return {
                "status": "success",
                "configuration": config
            }

        except exceptions.CosmosResourceNotFoundError:
            raise HTTPException(status_code=404, detail="Configuration not found")

    except Exception as e:
        print(f"Error getting metadata fields by ID: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get metadata fields: {str(e)}")

@app.delete("/api/metadata/{config_id}")
async def delete_metadata_fields(config_id: str, request: Request):
    """Delete a specific metadata fields configuration"""
    try:
        user_email = get_user_email(request)
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        try:
            metadata_fields_container.delete_item(item=config_id, partition_key=user_email)
            return {
                "status": "success",
                "message": "Metadata fields configuration deleted successfully"
            }
        except exceptions.CosmosResourceNotFoundError:
            raise HTTPException(status_code=404, detail="Configuration not found")

    except Exception as e:
        print(f"Error deleting metadata fields: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to delete metadata fields: {str(e)}")

@app.get("/api/config/{run_id}")
async def get_extraction_config(run_id: str, request: Request):
    """Get extraction configuration as text file for a specific run - Returns text directly without blob storage"""
    try:
        user_email = get_user_email(request)
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found")

        # Get the run record from Cosmos DB
        try:
            run_record = container.read_item(item=run_id, partition_key=user_email)
        except exceptions.CosmosResourceNotFoundError:
            raise HTTPException(status_code=404, detail="Run not found")

        # Build configuration text content
        config_lines = []
        config_lines.append("=" * 60)
        config_lines.append("EXTRACTION CONFIGURATION DETAILS")
        config_lines.append("=" * 60)
        config_lines.append("")

        # Basic Information
        config_lines.append("REPORT INFORMATION:")
        config_lines.append("-" * 20)
        config_lines.append(f"Report Name: {run_record.get('extractionReportName', 'Untitled Report')}")
        config_lines.append(f"User Email: {user_email}")
        config_lines.append(f"Extraction Date: {run_record.get('timestamp', 'Unknown')}")
        config_lines.append(f"Status: {run_record.get('status', 'Unknown')}")
        config_lines.append("")

        # Input Files Information
        config_lines.append("INPUT FILES:")
        config_lines.append("-" * 12)
        uploaded_files = run_record.get('uploadedFiles', [])
        if uploaded_files:
            for i, file_info in enumerate(uploaded_files, 1):
                config_lines.append(f"{i}. {file_info.get('fileName', 'Unknown file')}")
        else:
            config_lines.append("No uploaded files found")
        config_lines.append("")

        # Free Text Description
        config_lines.append("FREE TEXT DESCRIPTION:")
        config_lines.append("-" * 22)
        free_text_desc = run_record.get('freetextDescription', '')
        if free_text_desc:
            config_lines.append(free_text_desc)
        else:
            config_lines.append("No description provided")
        config_lines.append("")

        # Metadata Fields
        config_lines.append("METADATA FIELDS TO EXTRACT:")
        config_lines.append("-" * 28)
        metadata_fields = run_record.get('metadataFields', [])
        if metadata_fields:
            for i, field in enumerate(metadata_fields, 1):
                config_lines.append(f"{i}. Field Name: {field.get('name', 'Unknown')}")
                config_lines.append(f"   Description: {field.get('desc', 'No description')}")
                config_lines.append(f"   Classification: {'Yes' if field.get('classify', False) else 'No'}")
                
                # Add example data if available
                row_values = field.get('rowValues', {})
                if row_values:
                    config_lines.append("   Example Data:")
                    for row_key, row_value in list(row_values.items())[:3]:
                        if row_value and row_value.strip():
                            config_lines.append(f"     - {row_value}")
                config_lines.append("")
        else:
            config_lines.append("No metadata fields configured")
            config_lines.append("")

        # Extraction Results Summary
        extracted_data = run_record.get('extractedData', {})
        if extracted_data:
            table_data = extracted_data.get('tableData', [])
            config_lines.append("EXTRACTION RESULTS SUMMARY:")
            config_lines.append("-" * 28)
            config_lines.append(f"Total rows processed: {len(table_data)}")
            config_lines.append(f"Extraction timestamp: {extracted_data.get('extractionTimestamp', 'Unknown')}")
            
            if table_data and len(table_data) > 0:
                config_lines.append("")
                config_lines.append("Field Extraction Overview:")
                sample_row = table_data[0]
                for key in sample_row.keys():
                    if key not in ['Row Index', 'Original Text']:
                        if not key.endswith(' Confidence') and not key.endswith(' Reason'):
                            config_lines.append(f"  - {key}: Extracted")
            config_lines.append("")

        # Processing Statistics
        if 'processing_stats' in run_record:
            stats = run_record['processing_stats']
            config_lines.append("PROCESSING STATISTICS:")
            config_lines.append("-" * 22)
            config_lines.append(f"Processing time: {stats.get('total_time_seconds', 0):.2f} seconds")
            config_lines.append(f"Average time per row: {stats.get('avg_time_per_row', 0):.2f} seconds")
            config_lines.append(f"Rows per second: {stats.get('throughput_rows_per_second', 0):.2f}")
            
            # Instance strategy information
            instance_info = stats.get('instance_strategy', {})
            if instance_info:
                config_lines.append(f"Instance strategy: {instance_info.get('instances_used', 'Unknown')}")
                config_lines.append(f"Cost optimization: {instance_info.get('reasoning', 'N/A')}")
            config_lines.append("")

        # Footer
        config_lines.append("=" * 60)
        config_lines.append("END OF CONFIGURATION")
        config_lines.append("=" * 60)

        # Join all lines and create response
        config_text = "\n".join(config_lines)
        
        # Clean filename for header
        report_name = run_record.get('extractionReportName', 'report')
        clean_filename = re.sub(r'[<>:"/\\|?*]', '_', report_name).replace(' ', '_')

        # Return as text file response
        return Response(
            content=config_text,
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=config_{clean_filename}.txt"
            }
        )

    except Exception as e:
        print(f"Error getting extraction config: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get extraction config: {str(e)}")

@app.post("/api/extract-debug")
async def extract_debug(request: Request):
    """Debug endpoint to see what's being sent"""
    try:
        print(f"=== DEBUG ENDPOINT ===")
        print(f"Method: {request.method}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Content-Type: {request.headers.get('content-type', 'None')}")
        print(f"URL: {request.url}")

        # Try to get the form data
        try:
            form = await request.form()
            print(f"Form data keys: {list(form.keys())}")
            
            for key, value in form.items():
                print(f"Form field '{key}':")
                print(f"  Type: {type(value)}")
                print(f"  Has filename attr: {hasattr(value, 'filename')}")
                if hasattr(value, 'filename'):
                    print(f"  Filename: {value.filename}")
                    print(f"  Content-Type: {getattr(value, 'content_type', 'None')}")
                    print(f"  Size: {getattr(value, 'size', 'Unknown')} bytes")
                else:
                    print(f"  Value: {str(value)[:100]}...")

        except Exception as form_error:
            print(f"Error parsing form: {form_error}")

        # Try to get raw body
        try:
            body = await request.body()
            print(f"Raw body length: {len(body)} bytes")
            print(f"Raw body preview: {body[:300]}...")
        except Exception as body_error:
            print(f"Error reading body: {body_error}")

        print(f"=== END DEBUG ===")
        return {"status": "debug complete", "message": "Check server logs"}

    except Exception as e:
        print(f"Debug endpoint error: {e}")
        return {"error": str(e)}

@app.post("/api/extract")
async def extract_files_api(request: Request):
    """Handle file upload for metadata extraction from frontend using AI - Memory optimized"""
    extracted_results = []
    files = []
    metadata = ""
    df = None
    
    try:
        print(f"=== EXTRACT ENDPOINT REACHED ===")

        # Parse multipart form data
        try:
            form = await request.form()
            print(f"Successfully parsed form data")
            print(f"Form keys: {list(form.keys())}")

            # Extract files and metadata from form
            for key, value in form.items():
                print(f"Processing form field: {key}, type: {type(value)}, hasattr filename: {hasattr(value, 'filename')}")

                if hasattr(value, 'filename') and value.filename:
                    files.append(value)
                    print(f"Found file: {value.filename}")
                elif key == "files":
                    if hasattr(value, 'filename') and value.filename:
                        files.append(value)
                        print(f"Found file in files field: {value.filename}")
                elif key == "metadata":
                    metadata = str(value)
                    print(f"Found metadata: {len(metadata)} chars")
                else:
                    print(f"Other field: {key} = {str(value)[:50]}...")

        except Exception as form_error:
            print(f"Error parsing form: {form_error}")
            raise HTTPException(status_code=400, detail=f"Invalid form data: {form_error}")

        # If no files found in form, check if this might be a draft with file metadata
        if not files or len(files) == 0:
            print("No file uploads found, checking if this is a draft with file metadata...")

            try:
                if metadata:
                    metadata_obj = json.loads(metadata)
                    if "selectedFiles" in metadata_obj or "files" in metadata_obj:
                        file_info_list = metadata_obj.get("selectedFiles", []) or metadata_obj.get("files", [])
                        print(f"Found {len(file_info_list)} file references in metadata")

                        for i, file_info in enumerate(file_info_list):
                            print(f"Processing file reference {i}: {file_info}")

                            if isinstance(file_info, dict) and "blobPath" in file_info:
                                blob_path = file_info["blobPath"]
                                file_name = file_info.get("name", file_info.get("fileName", f"file_{i}"))

                                try:
                                    blob_client = blob_container_client.get_blob_client(blob_path)
                                    if blob_client.exists():
                                        blob_content = blob_client.download_blob().readall()
                                        mock_file = MockUploadFile(blob_content, file_name)
                                        files.append(mock_file)
                                        print(f"Successfully loaded file from blob: {file_name} ({len(blob_content)} bytes)")
                                    else:
                                        print(f"Warning: Blob not found for file {file_name}: {blob_path}")
                                        mock_file = MockUploadFile(b"", file_name)
                                        files.append(mock_file)

                                except Exception as blob_error:
                                    print(f"Error loading file {file_name} from blob storage: {str(blob_error)}")
                                    mock_file = MockUploadFile(b"", file_name)
                                    files.append(mock_file)
                                    continue

                            elif isinstance(file_info, dict) and ("content" in file_info or "data" in file_info):
                                file_name = file_info.get("name", file_info.get("fileName", f"file_{i}"))
                                content_field = "content" if "content" in file_info else "data"

                                try:
                                    content_data = file_info[content_field]
                                    if isinstance(content_data, str):
                                        if "base64," in content_data:
                                            content_data = content_data.split("base64,")[1]
                                        elif content_data.startswith("data:") and "," in content_data:
                                            content_data = content_data.split(",", 1)[1]

                                        file_content = base64.b64decode(content_data)
                                        mock_file = MockUploadFile(file_content, file_name)
                                        files.append(mock_file)
                                        print(f"Successfully loaded file from metadata content: {file_name} ({len(file_content)} bytes)")

                                except Exception as content_error:
                                    print(f"Error processing file content for {file_name}: {str(content_error)}")
                                    mock_file = MockUploadFile(b"", file_name)
                                    files.append(mock_file)
                                    continue

            except json.JSONDecodeError as json_error:
                print(f"Error parsing metadata JSON: {json_error}")

        # Validation and debug logging
        print(f"Files received: {len(files) if files else 0}")
        if files:
            for i, f in enumerate(files):
                file_type = "UploadFile" if hasattr(f, 'content_type') else "MockUploadFile"
                print(f"  File {i}: {f.filename} ({file_type})")
        print(f"Metadata received: {len(metadata) if metadata else 0} chars")

        if not metadata or metadata.strip() == "":
            print("ERROR: No metadata provided")
            raise HTTPException(status_code=400, detail="No metadata provided")

        metadata_obj = json.loads(metadata)
        print(f"Successfully parsed metadata JSON")

        # Check if this is a rerun/draft
        is_rerun = metadata_obj.get("reRun", False)
        print(f"Is rerun/draft: {is_rerun}")

        # Backend draft detection
        if not is_rerun:
            draft_indicators = [
                metadata_obj.get("id"),
                metadata_obj.get("selectedFiles") and len(metadata_obj.get("selectedFiles", [])) > 0,
                "fieldsData" in metadata_obj,
                metadata_obj.get("uploadedFiles") and len(metadata_obj.get("uploadedFiles", [])) > 0
            ]

            if any(draft_indicators):
                print(f"Backend detected this as a draft extraction")
                is_rerun = True

        if not files or len(files) == 0:
            print("ERROR: No files provided for extraction")
            raise HTTPException(status_code=400, detail="No files provided")

        # Print start time
        start_time = datetime.now()
        print(f"Task started at: {start_time}")

        # Extract metadata fields configuration
        if "fieldsData" in metadata_obj and isinstance(metadata_obj["fieldsData"], dict):
            # Loaded draft with nested structure
            fields_data = metadata_obj["fieldsData"]
            meta_data_fields = fields_data.get("metaDataFields", [])
            free_text_desc = fields_data.get("freeTextDesc", "")
            free_text_col_name = fields_data.get("freeTextColName", "")
            print(f"Processing loaded draft with nested fieldsData structure")

        elif "metaDataFields" in metadata_obj:
            # Direct structure
            meta_data_fields = metadata_obj.get("metaDataFields", [])
            free_text_desc = metadata_obj.get("freeTextDesc", "")
            free_text_col_name = metadata_obj.get("freeTextColName", "")
            print(f"Processing metadata with direct structure")

        else:
            # Fallback
            meta_data_fields = metadata_obj.get("metaDataFields", []) or metadata_obj.get("metadataFields", [])
            free_text_desc = metadata_obj.get("freeTextDesc", "") or metadata_obj.get("freetextDescription", "")
            free_text_col_name = metadata_obj.get("freeTextColName", "") or metadata_obj.get("freeTextColumn", "")
            print(f"Using fallback extraction")

        # Convert frontend fields to MetadataField objects
        ai_fields = []
        for field in meta_data_fields:
            field_name = normalize_field_name(field.get("name", ""))

            # Skip if field name matches input column name
            if field_name and free_text_col_name and field_name.upper() == free_text_col_name.upper():
                print(f"Skipping field '{field_name}' as it matches the input column name '{free_text_col_name}'")
                continue

            # Skip common input column indicators
            input_indicators = ['description', 'text', 'content', 'input', 'original', 'source']
            if any(indicator in field_name.lower() for indicator in input_indicators):
                if field_name.lower() in ['description', 'long description', 'text', 'content', 'input', 'original text', 'source']:
                    print(f"Skipping field '{field_name}' as it appears to be an input column name")
                    continue

            ai_field = MetadataField(
                name=field_name,
                description=field.get("desc", ""),
                isClassification=field.get("classify", False),
                classificationOptions=None
            )
            ai_fields.append(ai_field)

        print(f"Final AI fields to extract: {[f.name for f in ai_fields]}")

        # Extract examples for few-shot learning
        examples = []
        row_keys = set()
        for field in meta_data_fields:
            row_values = field.get("rowValues", {})
            row_keys.update(row_values.keys())

        for row_key in row_keys:
            example_values = {}
            example_text = ""

            for field in meta_data_fields:
                field_name = field.get("name", "")
                row_values = field.get("rowValues", {})
                if row_key in row_values:
                    field_value = row_values[row_key]
                    if field_value and field_value.strip():
                        example_values[field_name] = field_value

                if not example_text and field_name.lower() in ['text', 'description', 'content']:
                    example_text = field_value

            if example_values:
                if not example_text:
                    example_text = f"Example text for row {row_key}"

                example = ExampleRow(
                    text=example_text,
                    extracted_values=example_values
                )
                examples.append(example)

        print(f"Created {len(examples)} examples for few-shot learning")

        # Process each uploaded file with memory optimization
        for file_index, file in enumerate(files):
            print(f"Processing file {file_index + 1}/{len(files)}: {file.filename}")
            
            # Clear previous file's dataframe to free memory
            if df is not None:
                del df
                gc.collect()
            
            df = None
            if file.filename.endswith('.csv'):
                # Read CSV file with encoding handling
                content = await file.read()
                csv_content = None
                encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'windows-1252', 'iso-8859-1']
                for encoding in encodings_to_try:
                    try:
                        csv_content = content.decode(encoding)
                        print(f"Successfully decoded CSV file using {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue

                if csv_content is None:
                    csv_content = content.decode('utf-8', errors='ignore')
                    print(f"Warning: Used UTF-8 with error handling")

                # Parse CSV data with error handling
                try:
                    df = pd.read_csv(StringIO(csv_content))
                except pd.errors.ParserError as e:
                    print(f"Initial CSV parsing failed: {str(e)}")
                    try:
                        df = pd.read_csv(
                            StringIO(csv_content),
                            on_bad_lines='skip',
                            engine='python',
                            quoting=1
                        )
                        print(f"Successfully parsed CSV with error handling")
                    except Exception as fallback_error:
                        try:
                            df = pd.read_csv(
                                StringIO(csv_content),
                                sep=',',
                                quotechar='"',
                                skipinitialspace=True,
                                engine='python'
                            )
                            print(f"Successfully parsed CSV with fallback method")
                        except Exception as final_error:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Unable to parse CSV file {file.filename}. Please check for unescaped commas, quotes, or line breaks in your data."
                            )
                
                # Clear csv_content to free memory
                del csv_content
                del content
                gc.collect()

            elif file.filename.endswith(('.xlsx', '.xls')):
                # Read Excel file
                content = await file.read()
                try:
                    excel_buffer = BytesIO(content)
                    if file.filename.endswith('.xlsx'):
                        df = pd.read_excel(excel_buffer, engine='openpyxl')
                        print(f"Successfully parsed Excel (.xlsx) file")
                    else:
                        df = pd.read_excel(excel_buffer, engine='xlrd')
                        print(f"Successfully parsed Excel (.xls) file")
                except Exception as e:
                    try:
                        excel_buffer = BytesIO(content)
                        df = pd.read_excel(excel_buffer)
                        print(f"Successfully parsed Excel file with default engine")
                    except Exception as fallback_error:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unable to parse Excel file {file.filename}. Please ensure the file is a valid Excel format."
                        )
                
                # Clear content and excel_buffer to free memory
                del content
                del excel_buffer
                gc.collect()
            else:
                print(f"Skipping unsupported file type: {file.filename}")
                continue

            if df is None or len(df) == 0:
                print(f"Skipping {file.filename}: file is empty")
                continue

            # Find the text column
            text_column = None
            if free_text_col_name and free_text_col_name in df.columns:
                text_column = free_text_col_name
            else:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_column = col
                        break

            if text_column is None:
                continue

            print(f"Using text column: {text_column} for metadata extraction")

            # Apply row limit
            original_row_count = len(df)
            if len(df) > MAX_ROWS_LIMIT:
                print(f"Input file has {len(df)} rows, limiting to first {MAX_ROWS_LIMIT} rows")
                df = df.head(MAX_ROWS_LIMIT)

            print(f"Processing {len(df)} rows (original: {original_row_count})")

            # Cost optimization: Set up dynamic instance strategy
            instance_strategy = determine_instance_strategy(len(df))
            active_client_pool = setup_client_pool_for_strategy(instance_strategy)

            # Process rows in optimized batches - Memory efficient approach
            row_texts = []
            valid_indices = []
            for idx, row in df.iterrows():
                text = str(row[text_column]).strip()
                if not text or text.lower() in ['nan', 'none', '']:
                    continue
                row_texts.append(text)
                valid_indices.append(len(extracted_results) + len(row_texts) - 1)

            # Execute in optimized batches with semaphore limiting
            ai_results_list = []
            if row_texts:
                strategy_info = "DUAL" if instance_strategy["use_dual_instances"] else "SINGLE"
                print(f"Processing {len(row_texts)} rows in optimized batches using {strategy_info} instance strategy...")

                # Set concurrency limits based on strategy
                if instance_strategy["use_dual_instances"]:
                    concurrent_limit = 200  # Reduced from 300
                    batch_size = 200      # Reduced from 300
                else:
                    concurrent_limit = 100  # Reduced from 150
                    batch_size = 100      # Reduced from 150

                print(f"Using concurrent limit: {concurrent_limit}, batch size: {batch_size}")

                # Create semaphore to limit concurrent requests
                semaphore = asyncio.Semaphore(concurrent_limit)

                async def process_text_with_semaphore(text: str, row_index: int):
                    """Process a single text with semaphore limiting"""
                    async with semaphore:
                        return await extract_metadata_for_text_with_retry(
                            text, ai_fields, free_text_desc, examples=examples, row_index=row_index
                        )

                # Process in memory-efficient batches
                for batch_start in range(0, len(row_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(row_texts))
                    batch_texts = row_texts[batch_start:batch_end]
                    print(f"Processing batch {batch_start//batch_size + 1} ({strategy_info} mode): rows {batch_start} to {batch_end-1}")

                    # Create tasks for this batch
                    batch_tasks = []
                    for i, text in enumerate(batch_texts):
                        row_index = batch_start + i
                        task = process_text_with_semaphore(text, row_index)
                        batch_tasks.append(task)

                    # Process this batch concurrently
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results immediately to avoid memory buildup
                    for i, (text, ai_results) in enumerate(zip(batch_texts, batch_results)):
                        if isinstance(ai_results, Exception):
                            print(f"Exception processing row {batch_start + i}: {str(ai_results)}")
                            continue
                            
                        extracted_metadata = {}
                        for field_name, field_result in ai_results.items():
                            # Skip if field name matches input column name
                            if field_name.upper() == text_column.upper():
                                continue

                            # Skip if field name contains input column indicators
                            input_indicators = ['description', 'text', 'content', 'input', 'original']
                            if any(indicator in field_name.lower() for indicator in input_indicators):
                                if field_result.value.strip() == text.strip():
                                    continue

                            # Skip if value is identical to input text or too long
                            if field_result.value.strip() == text.strip() or len(field_result.value) > 150:
                                continue

                            extracted_metadata[field_name] = {
                                "value": field_result.value,
                                "confidence": field_result.confidence,
                                "reason": field_result.reason
                            }

                        result = {
                            "rowIndex": batch_start + i,
                            "originalText": text,
                            "extractedMetadata": extracted_metadata
                        }
                        extracted_results.append(result)

                    # Clear batch data to free memory
                    del batch_texts
                    del batch_tasks
                    del batch_results
                    gc.collect()

                    print(f"Batch completed. Progress: {len(extracted_results)} total results processed")

                    # Small delay between batches
                    if batch_start + batch_size < len(row_texts):
                        await asyncio.sleep(0.1)

            # Clear row processing data for this file
            del row_texts
            del valid_indices
            gc.collect()

        # Print completion time
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"Task started at: {start_time}")
        print(f"All processing completed at: {end_time}")
        print(f"Total processing time: {duration}")

        # Generate feedback summary - only if we have results
        feedback = None
        if extracted_results:
            try:
                feedback_results = []
                for result in extracted_results[:100]:  # Limit feedback generation to first 100 results
                    metadata_dict = {}
                    for field_name, field_data in result.get("extractedMetadata", {}).items():
                        metadata_dict[field_name] = MetadataFieldResult(
                            value=field_data.get("value", ""),
                            confidence=field_data.get("confidence", "Medium"),
                            reason=field_data.get("reason", "")
                        )

                    if metadata_dict:
                        feedback_result = MetadataExtractionResult(
                            rowIndex=result.get("rowIndex", len(feedback_results)),
                            originalText=result.get("originalText", ""),
                            extractedMetadata=metadata_dict
                        )
                        feedback_results.append(feedback_result)

                if feedback_results:
                    feedback = await generate_feedback_summary(feedback_results, ai_fields)
                    
                # Clear feedback data
                del feedback_results
                gc.collect()
            except Exception as feedback_error:
                print(f"Error generating feedback: {feedback_error}")

        # Save extraction results to blob storage and CosmosDB
        if extracted_results:
            try:
                report_name = metadata_obj.get("reportName", "Untitled Report")
                user_email = get_user_email(request) or metadata_obj.get("userEmail", "unknown@user.com")
                
                if user_email == "unknown@user.com":
                    print(f"Warning: Could not determine user email, using fallback")

                print(f"Using user email: {user_email}")

                # Process results into table format - Memory efficient
                processed_results = []
                batch_size = 1000  # Process in smaller batches to avoid memory issues
                
                for batch_start in range(0, len(extracted_results), batch_size):
                    batch_end = min(batch_start + batch_size, len(extracted_results))
                    batch_results = extracted_results[batch_start:batch_end]
                    
                    for result in batch_results:
                        processed_row = {
                            "Row Index": result["rowIndex"],
                            "Original Text": result["originalText"]
                        }

                        if result.get("extractedMetadata"):
                            for field_name, field_data in result["extractedMetadata"].items():
                                formatted_field_name = field_name.replace('_', ' ').title()
                                processed_row[formatted_field_name] = field_data.get("value", "")
                                processed_row[f"{formatted_field_name} Confidence"] = field_data.get("confidence", "Medium")
                                processed_row[f"{formatted_field_name} Reason"] = field_data.get("reason", "")

                        processed_results.append(processed_row)
                    
                    # Force garbage collection after each batch
                    gc.collect()

                # Upload files to blob storage
                files_blob_info = []
                for file in files:
                    try:
                        await file.seek(0)
                    except AttributeError:
                        pass

                    blob_name = f"{user_email}/{uuid.uuid4()}_{file.filename}"
                    data = await file.read()
                    blob_client = blob_container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(data, overwrite=True)

                    # Set expiration
                    expiry_time = datetime.utcnow() + timedelta(days=BLOB_EXPIRY_DAYS)
                    blob_client.set_blob_metadata({'expiry_time': expiry_time.isoformat()})

                    files_blob_info.append({
                        "fileName": file.filename,
                        "blobPath": blob_name,
                        "expiryTime": expiry_time.isoformat()
                    })

                # Create CSV file for extracted data
                extracted_csv_blob_path = None
                try:
                    df_results = pd.DataFrame(processed_results)
                    csv_buffer = StringIO()
                    df_results.to_csv(csv_buffer, index=False)
                    csv_content = csv_buffer.getvalue()

                    csv_blob_name = f"{user_email}/extracted_results_{uuid.uuid4()}.csv"
                    csv_blob_client = blob_container_client.get_blob_client(csv_blob_name)
                    csv_blob_client.upload_blob(csv_content.encode('utf-8'), overwrite=True)

                    expiry_time = datetime.utcnow() + timedelta(days=BLOB_EXPIRY_DAYS)
                    csv_blob_client.set_blob_metadata({'expiry_time': expiry_time.isoformat()})
                    extracted_csv_blob_path = csv_blob_name
                    
                    # Clear CSV data from memory
                    del df_results
                    del csv_buffer
                    del csv_content
                    gc.collect()

                except Exception as csv_error:
                    print(f"Warning: Failed to create CSV file: {str(csv_error)}")

                # Create Cosmos DB document
                run_id = str(uuid.uuid4())
                doc = {
                    "id": run_id,
                    "userEmail": user_email,
                    "timestamp": datetime.utcnow().isoformat(),
                    "expiryTime": (datetime.utcnow() + timedelta(days=BLOB_EXPIRY_DAYS)).isoformat(),
                    "extractionReportName": report_name,
                    "freetextDescription": free_text_desc,
                    "uploadedFiles": files_blob_info,
                    "metadataFields": meta_data_fields,
                    "expectingTableOutput": {"tableData": True},
                    "extractedData": {
                        "tableData": processed_results,
                        "extractionTimestamp": datetime.utcnow().isoformat()
                    },
                    "isDraft": False,
                    "isCompleted": True,
                    "status": "completed"
                }

                if extracted_csv_blob_path:
                    doc["extractedFile"] = {
                        "fileName": f"extracted_results_{report_name}.csv",
                        "blobPath": extracted_csv_blob_path,
                        "extractionTimestamp": datetime.utcnow().isoformat()
                    }

                container.create_item(body=doc)
                print(f"Successfully saved extraction results with run_id: {run_id}")

            except Exception as e:
                print(f"Warning: Failed to save extraction results: {str(e)}")

        print(f"Returning {len(extracted_results)} results")

        response = {
            "results": extracted_results,
            "feedback": feedback
        }

        # Add row limiting information if applicable
        if 'original_row_count' in locals() and original_row_count > MAX_ROWS_LIMIT:
            response["rowLimitInfo"] = {
                "originalRowCount": original_row_count,
                "processedRowCount": len(extracted_results),
                "wasLimited": True,
                "limit": MAX_ROWS_LIMIT,
                "message": f"Input file had {original_row_count} rows. Only the first {MAX_ROWS_LIMIT} rows were processed due to system limits."
            }

        return response

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in extract_files_api: {str(e)}")
        print(f"Received metadata string: {metadata[:500]}...")
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {str(e)}")

    except Exception as e:
        print(f"Error in extract_files_api: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    
    finally:
        # Cleanup memory in finally block
        try:
            if df is not None:
                del df
            if extracted_results:
                del extracted_results
            if files:
                del files
            if metadata:
                del metadata
            gc.collect()
            print("Memory cleanup completed")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/index.html")

app.mount("/", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    log.info("ChatENB WorkSmart AI WebApp Starting Up...")
    uvicorn.run(app, host="0.0.0.0", port=8000)