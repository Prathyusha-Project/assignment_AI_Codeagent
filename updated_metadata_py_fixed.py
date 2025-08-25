from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import uuid
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO
from openai import AsyncAzureOpenAI
import re
import tiktoken
import traceback
from collections import deque
import gc  # For garbage collection
import weakref  # For weak references to avoid memory leaks

# Shared constants
MAX_INPUT_TOKENS = 100000  # Maximum input tokens for GPT-4-turbo
MAX_OUTPUT_TOKENS = 16000  # Maximum output tokens
INSTRUCTION_TOKEN_RESERVE = 0.3  # Reserve 30% of tokens for instructions and overhead

# Rate limiting constants for GPT-4o - Dynamic instance optimization
MAX_REQUESTS_PER_MINUTE = 2700  # Per instance
MAX_TOKENS_PER_MINUTE = 450000  # Per instance

# Concurrency limits - Task 1: Reduced concurrent requests
MAX_CONCURRENT_REQUESTS = 200  # Reduced from 300 for dual instances
MAX_CONCURRENT_REQUESTS_SINGLE = 100  # Reduced from 150 for single instance

MAX_ROWS_LIMIT = 20000  # Maximum rows to process

# Dynamic instance thresholds - Cost optimization
SINGLE_INSTANCE_THRESHOLD = 1000  # Use single instance for <= 1000 rows
DUAL_INSTANCE_THRESHOLD = 1001   # Use dual instances for > 1000 rows
ESTIMATED_PROCESSING_TIME_PER_ROW = 0.05  # seconds per row for single instance

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [0.1, 0.3, 1.0]

router = APIRouter(prefix="/metadata", tags=["metadata"])

# Storage for jobs - in production, this should be a database with memory limits
jobs_storage: Dict[str, Any] = {}
MAX_JOBS_IN_MEMORY = 100  # Limit concurrent jobs to prevent memory issues

def cleanup_old_jobs():
    """Clean up old completed jobs to free memory"""
    if len(jobs_storage) <= MAX_JOBS_IN_MEMORY:
        return
        
    # Sort by creation time and remove oldest completed jobs
    job_items = [(job_id, job_data) for job_id, job_data in jobs_storage.items() 
                 if job_data.get("status") in ["completed", "error"]]
    job_items.sort(key=lambda x: x[1].get("created_at", datetime.min))
    
    # Remove oldest jobs beyond the limit
    jobs_to_remove = len(jobs_storage) - MAX_JOBS_IN_MEMORY + 10  # Remove extra to avoid frequent cleanup
    for i in range(min(jobs_to_remove, len(job_items))):
        job_id = job_items[i][0]
        del jobs_storage[job_id]
        print(f"Cleaned up old job: {job_id}")

def get_token_count(text: str) -> int:
    """Count the number of tokens in a text string using tiktoken"""
    try:
        encoder = tiktoken.encoding_for_model("gpt-4")
        return len(encoder.encode(text))
    except Exception:
        return len(text) // 4  # Rough approximation

def determine_instance_strategy(row_count: int) -> dict:
    """Determine the optimal instance strategy based on row count for cost optimization"""
    strategy = {
        "use_dual_instances": False,
        "primary_only": True,
        "estimated_time_minutes": 0,
        "reasoning": ""
    }
   
    if row_count <= SINGLE_INSTANCE_THRESHOLD:
        strategy.update({
            "use_dual_instances": False,
            "primary_only": True,
            "estimated_time_minutes": (row_count * ESTIMATED_PROCESSING_TIME_PER_ROW) / 60,
            "reasoning": f"Using single instance for {row_count} rows (cost-effective for ≤{SINGLE_INSTANCE_THRESHOLD} rows)"
        })
    else:
        strategy.update({
            "use_dual_instances": True,
            "primary_only": False,
            "estimated_time_minutes": (row_count * ESTIMATED_PROCESSING_TIME_PER_ROW) / 120,
            "reasoning": f"Using dual instances for {row_count} rows (optimal for >{SINGLE_INSTANCE_THRESHOLD} rows)"
        })
   
    return strategy

class RateLimiter:
    """Rate limiter to handle GPT-4o API limits with dual instance support"""
    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int, instance_name: str = "default"):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.instance_name = instance_name
        self.request_times = deque(maxlen=1000)  # Limit deque size to prevent memory growth
        self.token_usage = deque(maxlen=1000)   # Limit deque size
        self.lock = asyncio.Lock()
   
    async def acquire(self, estimated_tokens: int = 500):
        """Acquire permission to make a request with estimated token usage"""
        async with self.lock:
            now = time.time()
            minute_ago = now - 60
           
            # Remove old entries - more efficient cleanup
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
           
            while self.token_usage and self.token_usage[0][0] < minute_ago:
                self.token_usage.popleft()
           
            # Calculate current usage
            current_requests = len(self.request_times)
            current_tokens = sum(usage[1] for usage in self.token_usage)
           
            # Use 90% of limits - more aggressive with dual instances
            requests_threshold = int(self.max_requests_per_minute * 0.9)
            tokens_threshold = int(self.max_tokens_per_minute * 0.9)
           
            # Only wait if we're really close to limits
            if current_requests >= requests_threshold or current_tokens + estimated_tokens > tokens_threshold:
                if current_requests >= requests_threshold and self.request_times:
                    wait_time = max(0.05, min(0.5, self.request_times[0] + 60 - now))
                elif current_tokens + estimated_tokens > tokens_threshold and self.token_usage:
                    wait_time = max(0.05, min(0.5, self.token_usage[0][0] + 60 - now))
                else:
                    wait_time = 0.05
               
                wait_time = min(wait_time, 0.5)
               
                if wait_time > 0:
                    print(f"[{self.instance_name}] Rate limit approached. Brief wait {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
           
            # Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))

# Global rate limiter instances
rate_limiter_1 = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE, "Instance-1")
rate_limiter_2 = RateLimiter(MAX_REQUESTS_PER_MINUTE, MAX_TOKENS_PER_MINUTE, "Instance-2")

# Global instance strategy tracking
current_instance_strategy = {
    "use_dual_instances": False,
    "primary_only": True,
    "active_client_pool": []
}

# Initialize dual Azure OpenAI clients
try:
    # Primary instance
    client_1 = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_SERVICE_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2023-05-15"
    )
    deployment_1 = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    print(f"Instance 1 initialized: {deployment_1}")
   
    # Secondary instance
    client_2 = AsyncAzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_SERVICE_KEY_2"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_2"),
        api_version="2023-05-15"
    )
    deployment_2 = os.getenv("AZURE_OPENAI_DEPLOYMENT_2", "gpt-4o")
    print(f"Instance 2 initialized: {deployment_2}")
   
    # Client pool for load balancing
    client_pool = [
        (client_1, deployment_1, rate_limiter_1),
        (client_2, deployment_2, rate_limiter_2)
    ]
   
except Exception as e:
    print(f"Warning: Could not initialize OpenAI clients: {e}")
    client_1 = None
    client_2 = None
    deployment_1 = None
    deployment_2 = None
    client_pool = []

def setup_client_pool_for_strategy(strategy: dict) -> list:
    """Setup client pool based on the determined strategy"""
    global current_instance_strategy
   
    if strategy["use_dual_instances"]:
        active_pool = [
            (client_1, deployment_1, rate_limiter_1),
            (client_2, deployment_2, rate_limiter_2)
        ]
        print(f"💰 Cost Optimization: Using DUAL instances for large dataset")
        print(f"   Expected time: ~{strategy['estimated_time_minutes']:.1f} minutes")
        print(f"   Reasoning: {strategy['reasoning']}")
    else:
        active_pool = [
            (client_1, deployment_1, rate_limiter_1)
        ]
        print(f"💰 Cost Optimization: Using SINGLE instance for small dataset")
        print(f"   Expected time: ~{strategy['estimated_time_minutes']:.1f} minutes")
        print(f"   Reasoning: {strategy['reasoning']}")
   
    current_instance_strategy.update(strategy)
    current_instance_strategy["active_client_pool"] = active_pool
   
    return active_pool

# Pydantic models
class MetadataField(BaseModel):
    name: str
    description: str
    isClassification: bool
    classificationOptions: Optional[List[str]] = None

class ExampleRow(BaseModel):
    text: str
    extracted_values: Dict[str, str]

class MetadataExtractionConfig(BaseModel):
    freeTextColumnIndex: int
    freeTextColumnName: Optional[str] = None
    freeTextColumnDescription: Optional[str] = ""
    metadataFields: List[MetadataField]
    exampleRows: Optional[List[int]] = []

class MetadataExtractionRequest(BaseModel):
    config: MetadataExtractionConfig
    csvData: List[Dict[str, Any]]

class MetadataFieldResult(BaseModel):
    value: str
    confidence: str  # "High", "Medium", or "Low"
    reason: str

class MetadataExtractionResult(BaseModel):
    rowIndex: int
    originalText: str
    extractedMetadata: Dict[str, MetadataFieldResult]

class ProcessingStats(BaseModel):
    total_rows: int
    total_time_seconds: float
    avg_time_per_row: float
    start_time: str
    end_time: str

class MetadataExtractionResponse(BaseModel):
    jobId: str
    status: str
    progress: int
    results: Optional[List[MetadataExtractionResult]] = None
    error: Optional[str] = None
    processing_stats: Optional[ProcessingStats] = None
    feedback: Optional[str] = None

class ReprocessRequest(BaseModel):
    rowIndices: List[int]

async def extract_metadata_for_text_with_retry(text: str, fields: List[MetadataField], column_description: str = "", examples: Optional[List[ExampleRow]] = None, row_index: int = -1) -> Dict[str, MetadataFieldResult]:
    """Extract metadata from a single text using OpenAI with retry logic and dynamic instance load balancing - Memory optimized"""
   
    # Initialize results for all fields
    results = {}
    for field in fields:
        results[field.name] = MetadataFieldResult(
            value="Unknown",
            confidence="Low",
            reason="Initial placeholder - processing not yet attempted"
        )
   
    if not current_instance_strategy.get("active_client_pool") and not client_pool:
        for field in fields:
            results[field.name] = MetadataFieldResult(
                value="OpenAI client not available",
                confidence="Low",
                reason="OpenAI clients could not be initialized"
            )
        return results

    # Validate input text
    if not text or not text.strip():
        for field in fields:
            results[field.name] = MetadataFieldResult(
                value="null",
                confidence="Low",
                reason="No text provided for extraction"
            )
        return results

    def is_input_column_name(field_name: str) -> bool:
        """Check if a field name represents an input column rather than extracted metadata"""
        input_indicators = [
            'description', 'text', 'content', 'input', 'original', 'source',
            'long description', 'long_description', 'longdescription',
            'full text', 'full_text', 'fulltext', 'raw text', 'raw_text',
            'narrative', 'details', 'information', 'data'
        ]
        field_lower = field_name.lower().strip()
        return any(indicator in field_lower for indicator in input_indicators)

    # Dynamic load balancing based on current strategy
    active_pool = current_instance_strategy.get("active_client_pool", [(client_1, deployment_1, rate_limiter_1)])
   
    if not active_pool:
        active_pool = [(client_1, deployment_1, rate_limiter_1)]
   
    if len(active_pool) == 1:
        client_instance = active_pool[0]
    else:
        instance_index = 0 if row_index % 2 == 0 else min(1, len(active_pool) - 1)
        client_instance = active_pool[instance_index]
   
    client, deployment, rate_limiter = client_instance
   
    # Retry logic with dual instance fallback
    for attempt in range(MAX_RETRIES):
        try:
            # Truncate text if too long to prevent memory issues
            max_text_length = 8000  # Reduced from unlimited to prevent memory issues
            truncated_text = text[:max_text_length] if len(text) > max_text_length else text
            
            estimated_tokens = get_token_count(truncated_text[:1000]) + len(fields) * 20 + 150
           
            await rate_limiter.acquire(estimated_tokens)
           
            # Single API call combining extraction and confidence assessment
            messages = [
                {
                    "role": "system",
                    "content": """You are a metadata extraction assistant. For each field requested, you need to:

1. Extract the most relevant value from the text
2. Assess your confidence in that extraction (High, Medium, or Low)
3. Provide a brief reason for your confidence level

IMPORTANT RULES:
- If you are unsure about how to extract a given field, assign it "unknown"
- If you do not think the field is in the freetext, assign it "null"
- Always provide a clear reason explaining your decision
- Never leave any field empty or undefined
- Keep extracted values concise (under 100 characters when possible)

Return a JSON object where each field name maps to an object with 'value', 'confidence', and 'reason' keys.

Example format:
{
  "field_name": {
    "value": "extracted_value",
    "confidence": "High",
    "reason": "Clear match found in text"
  },
  "other_field": {
    "value": "null",
    "confidence": "Low",
    "reason": "Field not present in the provided text"
  }
}"""
                },
                {
                    "role": "user",
                    "content": f"Text: {truncated_text}\n\nColumn context: {column_description}\n\nExtract the following fields with confidence assessment:\n"
                }
            ]
           
            # Add examples for few-shot learning if provided - limit to prevent memory issues
            if examples and len(examples) > 0:
                example_content = "\nHere are some examples:\n\n"
                # Limit examples to first 3 to prevent memory bloat
                limited_examples = examples[:3]
                for i, example in enumerate(limited_examples):
                    example_content += f"Example {i+1}:\n"
                    example_content += f"Text: {example.text[:200]}...\n"  # Truncate example text
                    example_content += f"Extracted: {json.dumps(example.extracted_values)}\n\n"
               
                messages[1]["content"] = example_content + messages[1]["content"]
           
            # Add field extraction instructions
            for field in fields:
                if field.isClassification and field.classificationOptions:
                    instruction = f"- {field.name}: {field.description}. Choose from: {', '.join(field.classificationOptions[:5])}"  # Limit options
                else:
                    instruction = f"- {field.name}: {field.description}"
                messages[-1]["content"] += instruction + "\n"
           
            messages[-1]["content"] += "\nReturn JSON object with value, confidence, and reason for each field."
           
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800,  # Reduced from 1000 to limit memory usage
                ),
                timeout=25.0  # Reduced timeout to fail faster
            )
           
            response_content = response.choices[0].message.content
            if row_index % 100 == 0:  # Only log every 100th response to reduce memory
                print(f"API response received for row {row_index}: {len(response_content)} characters")
           
            # Clean response content
            if response_content.strip().startswith("```"):
                response_content = re.sub(r"^```[a-zA-Z]*\n?", "", response_content.strip())
                response_content = re.sub(r"```$", "", response_content.strip())
           
            json_match = re.search(r'{[\s\S]*}', response_content)
            if json_match:
                response_content = json_match.group(0)
           
            extracted_data = json.loads(response_content)
           
            # Process each field
            for field in fields:
                if is_input_column_name(field.name):
                    continue
               
                field_data = extracted_data.get(field.name, {})
               
                if isinstance(field_data, str):
                    field_data = {
                        "value": field_data,
                        "confidence": "Medium",
                        "reason": "Value extracted, confidence auto-assigned"
                    }
                elif not isinstance(field_data, dict):
                    field_data = {
                        "value": str(field_data) if field_data is not None else "unknown",
                        "confidence": "Low",
                        "reason": "Unexpected response format"
                    }
               
                extracted_value = field_data.get("value", "unknown")
                confidence = field_data.get("confidence", "Medium")
                reason = field_data.get("reason", "No reason provided")
               
                if confidence not in ["High", "Medium", "Low"]:
                    confidence = "Medium"
               
                # Truncate long values to prevent memory bloat
                if isinstance(extracted_value, str) and len(extracted_value) > 200:
                    extracted_value = extracted_value[:200] + "..."
                
                if isinstance(reason, str) and len(reason) > 100:
                    reason = reason[:100] + "..."
               
                if not extracted_value or extracted_value in ["", "Unable to extract", "Not found"]:
                    extracted_value = "null"
                    reason = "Field not found in the provided text"
                    confidence = "Low"
                elif extracted_value.lower() in ["unsure", "uncertain", "unclear"]:
                    extracted_value = "unknown"
                    reason = "Unable to determine field value with confidence"
                    confidence = "Low"
               
                # Check for input text duplication
                if extracted_value == truncated_text or (isinstance(extracted_value, str) and extracted_value.strip() == truncated_text.strip()):
                    extracted_value = "null"
                    confidence = "Low"
                    reason = "Extracted value was identical to input text"
               
                results[field.name] = MetadataFieldResult(
                    value=str(extracted_value),
                    confidence=confidence,
                    reason=str(reason)
                )
           
            # Clear variables to free memory
            del messages
            del response
            del response_content
            del extracted_data
            gc.collect()
            
            return results
           
        except asyncio.TimeoutError:
            print(f"Timeout error for row {row_index}, attempt {attempt + 1}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAYS[attempt])
                continue
            else:
                for field in fields:
                    if not is_input_column_name(field.name):
                        results[field.name] = MetadataFieldResult(
                            value="unknown",
                            confidence="Low",
                            reason=f"API timeout after {MAX_RETRIES} attempts"
                        )
                return results
       
        except json.JSONDecodeError as e:
            print(f"JSON decode error for row {row_index}, attempt {attempt + 1}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAYS[attempt])
                continue
            else:
                for field in fields:
                    if not is_input_column_name(field.name):
                        results[field.name] = MetadataFieldResult(
                            value="unknown",
                            confidence="Low",
                            reason=f"JSON parsing failed after {MAX_RETRIES} attempts"
                        )
                return results
       
        except Exception as e:
            print(f"Extraction error for row {row_index}, attempt {attempt + 1}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAYS[attempt])
                continue
            else:
                for field in fields:
                    if not is_input_column_name(field.name):
                        results[field.name] = MetadataFieldResult(
                            value="unknown",
                            confidence="Low",
                            reason=f"Extraction failed after {MAX_RETRIES} attempts: {str(e)[:100]}"
                        )
                return results
   
    return results

# Keep original function name for backward compatibility
async def extract_metadata_for_text(text: str, fields: List[MetadataField], column_description: str = "", examples: Optional[List[ExampleRow]] = None) -> Dict[str, MetadataFieldResult]:
    """Extract metadata from a single text using OpenAI with combined extraction and confidence assessment"""
    return await extract_metadata_for_text_with_retry(text, fields, column_description, examples, -1)

async def generate_feedback_summary(results: List[MetadataExtractionResult], fields: List[MetadataField]) -> Optional[str]:
    """Generate user-friendly feedback based on confidence scores and reasons - Memory optimized"""
   
    if not client_pool or not results:
        return None
   
    client, deployment, _ = client_pool[0]
   
    # Limit analysis to first 50 results to prevent memory issues
    limited_results = results[:50]
   
    # Analyze confidence distribution
    total_extractions = len(limited_results) * len(fields)
    low_confidence_count = 0
    medium_confidence_count = 0
    high_confidence_count = 0
   
    confidence_issues = []
    field_issues = {}
   
    for result in limited_results:
        for field_name, field_result in result.extractedMetadata.items():
            confidence = field_result.confidence.lower()
           
            if confidence == "low":
                low_confidence_count += 1
                confidence_issues.append({
                    "field": field_name,
                    "reason": field_result.reason[:100],  # Truncate reason
                    "text_sample": result.originalText[:50] + "..." if len(result.originalText) > 50 else result.originalText
                })
               
                if field_name not in field_issues:
                    field_issues[field_name] = {"low": 0, "medium": 0, "reasons": []}
                field_issues[field_name]["low"] += 1
                field_issues[field_name]["reasons"].append(field_result.reason[:50])  # Truncate
               
            elif confidence == "medium":
                medium_confidence_count += 1
                if field_name not in field_issues:
                    field_issues[field_name] = {"low": 0, "medium": 0, "reasons": []}
                field_issues[field_name]["medium"] += 1
                field_issues[field_name]["reasons"].append(field_result.reason[:50])  # Truncate
               
            elif confidence == "high":
                high_confidence_count += 1
   
    # Calculate percentages
    low_percentage = (low_confidence_count / total_extractions) * 100 if total_extractions > 0 else 0
    medium_percentage = (medium_confidence_count / total_extractions) * 100 if total_extractions > 0 else 0
    high_percentage = (high_confidence_count / total_extractions) * 100 if total_extractions > 0 else 0
   
    # Only generate feedback if there are medium or low confidence results
    if low_percentage < 5 and medium_percentage < 10:
        return None  # High accuracy, no feedback needed
   
    try:
        field_descriptions = "\n".join([f"- {field.name}: {field.description[:100]}" for field in fields[:10]])  # Limit fields and truncate descriptions
       
        field_analysis = ""
        for field_name, issues in list(field_issues.items())[:5]:  # Limit to 5 fields
            if issues["low"] > 0 or issues["medium"] > 0:
                field_analysis += f"\nField '{field_name}':\n"
                field_analysis += f"  - Low confidence: {issues['low']} instances\n"
                field_analysis += f"  - Medium confidence: {issues['medium']} instances\n"
               
                unique_reasons = list(set(issues["reasons"]))[:2]  # Limit to 2 unique reasons
                field_analysis += f"  - Common issues: {', '.join(unique_reasons)}\n"
       
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides actionable feedback for improving metadata extraction. "
                    "Based on the extraction statistics and common issues, provide clear, concise recommendations "
                    "in a user-friendly format. Focus on practical steps users can take to improve their results. "
                    "Keep response under 250 words."
                )
            },
            {
                "role": "user",
                "content": f"""

Metadata Extraction Analysis (Sample of {len(limited_results)} results):

Extraction Statistics:
- Total extractions: {total_extractions}
- High confidence: {high_confidence_count} ({high_percentage:.1f}%)
- Medium confidence: {medium_confidence_count} ({medium_percentage:.1f}%)
- Low confidence: {low_confidence_count} ({low_percentage:.1f}%)

Field Definitions:
{field_descriptions}

Field-Specific Issues:
{field_analysis}

Sample Issues:
{chr(10).join([f"- {issue['field']}: {issue['reason']}" for issue in confidence_issues[:3]])}

Please provide user-friendly feedback with specific steps to improve extraction accuracy.

Format your response as actionable recommendations, focusing on:
1. Data quality improvements
2. Field definition clarity
3. Example enhancements
4. Text preprocessing suggestions

Keep the response concise but helpful (max 250 words).

Use plain text format without markdown symbols, headers, or asterisks for emphasis.

                """
            }
        ]
       
        feedback_response = await asyncio.wait_for(
            client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0.3,
                max_tokens=300,  # Reduced token limit
            ),
            timeout=15.0  # Reduced timeout
        )
       
        feedback = feedback_response.choices[0].message.content.strip()
       
        # Clean up markdown formatting
        feedback = re.sub(r'^#{1,6}\s*', '', feedback, flags=re.MULTILINE)
        feedback = re.sub(r'\*\*([^\*]+)\*\*', r'\1', feedback)
        feedback = re.sub(r'\*([^\*]+)\*', r'\1', feedback)
        feedback = re.sub(r'^\*+\s*', '- ', feedback, flags=re.MULTILINE)
        feedback = re.sub(r'\n\s*\n', '\n\n', feedback)
        feedback = feedback.strip()
        
        # Truncate if still too long
        if len(feedback) > 2000:
            feedback = feedback[:2000] + "..."
       
        # Clear variables
        del messages
        del feedback_response
        del limited_results
        gc.collect()
       
        return feedback
       
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return None

async def process_single_row(row_index: int, row_data: Dict[str, Any], free_text_column: str, fields: List[MetadataField], column_description: str) -> MetadataExtractionResult:
    """Process a single row with proper error handling and retry logic - Memory optimized"""
    try:
        text = str(row_data.get(free_text_column, "")).strip()
        
        # Truncate text if too long
        if len(text) > 10000:
            text = text[:10000]
       
        if text:
            metadata = await extract_metadata_for_text_with_retry(
                text,
                fields,
                column_description,
                row_index=row_index
            )
           
            if not metadata or len(metadata) == 0:
                metadata = {}
                for field in fields:
                    metadata[field.name] = MetadataFieldResult(
                        value="unknown",
                        confidence="Low",
                        reason="No metadata was returned from extraction function"
                    )
            else:
                for field in fields:
                    if field.name not in metadata:
                        metadata[field.name] = MetadataFieldResult(
                            value="unknown",
                            confidence="Low",
                            reason="Field was missing from extraction results"
                        )
        else:
            metadata = {}
            for field in fields:
                metadata[field.name] = MetadataFieldResult(
                    value="null",
                    confidence="Low",
                    reason="No text provided for extraction - text field is empty or null"
                )
       
        result = MetadataExtractionResult(
            rowIndex=row_index,
            originalText=text if text else "",
            extractedMetadata=metadata
        )
       
        return result
       
    except Exception as e:
        print(f"Critical error processing row {row_index}: {str(e)}")
       
        fallback_metadata = {}
        for field in fields:
            fallback_metadata[field.name] = MetadataFieldResult(
                value="Error",
                confidence="Low",
                reason=f"Critical processing error: {str(e)[:50]}"
            )
       
        fallback_result = MetadataExtractionResult(
            rowIndex=row_index,
            originalText=str(row_data.get(free_text_column, ""))[:100] if row_data else "",
            extractedMetadata=fallback_metadata
        )
       
        return fallback_result

async def process_metadata_extraction(job_id: str, request: MetadataExtractionRequest):
    """Background task to process metadata extraction with cost-optimized dynamic instance selection - Memory optimized"""
    try:
        start_time = datetime.now()
        jobs_storage[job_id]["status"] = "processing"
        jobs_storage[job_id]["progress"] = 0
       
        csv_data = request.csvData
        config = request.config
       
        if not csv_data:
            raise ValueError("No CSV data provided")
       
        # Limit rows if necessary
        original_row_count = len(csv_data)
        if len(csv_data) > MAX_ROWS_LIMIT:
            print(f"Input has {len(csv_data)} rows, limiting to first {MAX_ROWS_LIMIT} rows")
            csv_data = csv_data[:MAX_ROWS_LIMIT]
       
        column_names = list(csv_data[0].keys()) if csv_data else []
       
        # Determine the free text column
        free_text_column = None
       
        if config.freeTextColumnName and config.freeTextColumnName in column_names:
            free_text_column = config.freeTextColumnName
            print(f"Using specified column name: {free_text_column}")
        elif config.freeTextColumnIndex < len(column_names):
            free_text_column = column_names[config.freeTextColumnIndex]
            print(f"Using column at index {config.freeTextColumnIndex}: {free_text_column}")
        else:
            raise ValueError("Invalid column index or name. No valid column found for text extraction.")
       
        total_rows = len(csv_data)
        print(f"Starting extraction of {total_rows} rows...")
       
        if original_row_count > MAX_ROWS_LIMIT:
            print(f"Note: Processing first {total_rows} of {original_row_count} total rows")
       
        # Cost optimization: Determine optimal instance strategy
        instance_strategy = determine_instance_strategy(total_rows)
        active_client_pool = setup_client_pool_for_strategy(instance_strategy)
       
        # Set concurrent limits based on strategy
        if instance_strategy["use_dual_instances"]:
            concurrent_limit = MAX_CONCURRENT_REQUESTS  # 200
            batch_size = min(200, concurrent_limit)
        else:
            concurrent_limit = MAX_CONCURRENT_REQUESTS_SINGLE  # 100
            batch_size = min(100, concurrent_limit)
       
        print(f"Using concurrent limit: {concurrent_limit}, batch size: {batch_size}")
       
        semaphore = asyncio.Semaphore(concurrent_limit)
       
        async def process_row_with_semaphore(row_index: int, row_data: Dict[str, Any]) -> MetadataExtractionResult:
            """Process a single row with semaphore limiting"""
            async with semaphore:
                return await process_single_row(
                    row_index,
                    row_data,
                    free_text_column,
                    config.metadataFields,
                    config.freeTextColumnDescription
                )
       
        # Process rows in optimized batches with memory management
        results = []
        processed_count = 0
        error_count = 0
        empty_count = 0
       
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_rows = csv_data[batch_start:batch_end]
           
            instance_info = "DUAL" if instance_strategy["use_dual_instances"] else "SINGLE"
            print(f"📦 Processing batch {batch_start//batch_size + 1} ({instance_info} mode): rows {batch_start} to {batch_end-1}")
           
            batch_tasks = []
            for i, row in enumerate(batch_rows):
                row_index = batch_start + i
                task = process_row_with_semaphore(row_index, row)
                batch_tasks.append(task)
           
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
           
            for i, result in enumerate(batch_results):
                row_index = batch_start + i
               
                if isinstance(result, Exception):
                    print(f"Exception in batch processing for row {row_index}: {str(result)}")
                    fallback_metadata = {}
                    for field in config.metadataFields:
                        fallback_metadata[field.name] = MetadataFieldResult(
                            value="Error",
                            confidence="Low",
                            reason=f"Batch processing error: {str(result)[:50]}"
                        )
                   
                    error_result = MetadataExtractionResult(
                        rowIndex=row_index,
                        originalText=str(batch_rows[i].get(free_text_column, ""))[:100] if i < len(batch_rows) else "",
                        extractedMetadata=fallback_metadata
                    )
                    results.append(error_result)
                    error_count += 1
                else:
                    results.append(result)
                   
                    text = str(batch_rows[i].get(free_text_column, "")).strip() if i < len(batch_rows) else ""
                    if not text:
                        empty_count += 1
                    else:
                        successful = any(
                            field_result.value not in ["Error", "unknown", "null"]
                            for field_result in result.extractedMetadata.values()
                        )
                        if successful:
                            processed_count += 1
                        else:
                            error_count += 1
           
            # Update progress after each batch
            progress = int((len(results) / total_rows) * 100)
            jobs_storage[job_id]["progress"] = progress
           
            print(f"Batch completed. Progress: {progress}% ({len(results)}/{total_rows} rows)")
           
            # Memory cleanup after each batch
            del batch_tasks
            del batch_results
            del batch_rows
            gc.collect()
           
            # Small delay between batches
            if batch_start + batch_size < total_rows:
                await asyncio.sleep(0.1)
       
        # Verify we processed all rows
        if len(results) != total_rows:
            print(f"WARNING: Expected {total_rows} results but got {len(results)}")
            for missing_index in range(len(results), total_rows):
                fallback_metadata = {}
                for field in config.metadataFields:
                    fallback_metadata[field.name] = MetadataFieldResult(
                        value="Error",
                        confidence="Low",
                        reason="Row was not processed due to system error"
                    )
               
                missing_result = MetadataExtractionResult(
                    rowIndex=missing_index,
                    originalText="",
                    extractedMetadata=fallback_metadata
                )
                results.append(missing_result)
       
        end_time = datetime.now()
        processing_time = end_time - start_time
       
        total_seconds = processing_time.total_seconds()
        avg_time_per_row = total_seconds / total_rows if total_rows > 0 else 0
       
        print(f"\nFinal Extraction Statistics:")
        print(f"💰 Instance Strategy: {instance_strategy['reasoning']}")
        print(f"   Instances Used: {'DUAL' if instance_strategy['use_dual_instances'] else 'SINGLE'}")
        print(f"   Estimated vs Actual Time: {instance_strategy['estimated_time_minutes']:.1f} min (est) vs {total_seconds/60:.1f} min (actual)")
        print(f"Total input rows: {total_rows}")
        print(f"Total output results: {len(results)}")
        print(f"Successfully processed: {processed_count}")
        print(f"Empty text rows: {empty_count}")
        print(f"Error rows: {error_count}")
        print(f"Processing time: {processing_time}")
        print(f"Average time per row: {avg_time_per_row:.2f} seconds")
        print(f"Throughput: {total_rows / total_seconds:.2f} rows/second")
        print(f"Extraction completed at: {end_time}")
       
        if original_row_count > MAX_ROWS_LIMIT:
            print(f"Note: Processed {total_rows} of {original_row_count} original rows")
       
        if len(results) == total_rows:
            print("✅ All rows have been processed and included in results")
        else:
            print(f"⚠️ Mismatch: Expected {total_rows} but got {len(results)} results")
       
        # Generate feedback summary with memory optimization
        feedback = None
        try:
            feedback = await generate_feedback_summary(results, config.metadataFields)
        except Exception as feedback_error:
            print(f"Error generating feedback: {feedback_error}")
       
        # Final validation - limited to prevent memory issues
        final_validation_errors = []
        if len(results) != total_rows:
            final_validation_errors.append(f"Result count mismatch: expected {total_rows}, got {len(results)}")
       
        expected_field_names = [field.name for field in config.metadataFields]
        for i, result in enumerate(results[:10]):  # Only validate first 10
            missing_fields = []
            for field_name in expected_field_names:
                if field_name not in result.extractedMetadata:
                    missing_fields.append(field_name)
            if missing_fields:
                final_validation_errors.append(f"Row {i} missing fields: {missing_fields}")
       
        if final_validation_errors:
            print("Final validation errors found:")
            for error in final_validation_errors[:5]:  # Limit error reporting
                print(f"  - {error}")
        else:
            print("✅ Final validation passed - all rows have complete metadata")
       
        jobs_storage[job_id].update({
            "status": "completed",
            "results": results,
            "progress": 100,
            "processing_stats": {
                "total_rows": total_rows,
                "original_row_count": original_row_count,
                "processed_count": processed_count,
                "empty_count": empty_count,
                "error_count": error_count,
                "final_result_count": len(results),
                "validation_errors": len(final_validation_errors),
                "total_time_seconds": total_seconds,
                "avg_time_per_row": avg_time_per_row,
                "throughput_rows_per_second": total_rows / total_seconds if total_seconds > 0 else 0,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "rows_limited": original_row_count > MAX_ROWS_LIMIT,
                "instance_strategy": {
                    "use_dual_instances": instance_strategy["use_dual_instances"],
                    "instances_used": "DUAL" if instance_strategy["use_dual_instances"] else "SINGLE",
                    "reasoning": instance_strategy["reasoning"],
                    "estimated_time_minutes": instance_strategy["estimated_time_minutes"],
                    "actual_time_minutes": total_seconds / 60,
                    "cost_optimization_applied": True
                }
            },
            "feedback": feedback
        })
       
        print(f"Job {job_id} completed successfully with {len(results)} results")
        
        # Clean up memory
        del csv_data
        del results
        gc.collect()
        cleanup_old_jobs()  # Clean up old jobs to free memory
       
    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        traceback.print_exc()
        jobs_storage[job_id]["status"] = "error"
        jobs_storage[job_id]["error"] = str(e)
        # Clean up on error
        gc.collect()

@router.post("/extract", response_model=MetadataExtractionResponse)
async def start_metadata_extraction(request: MetadataExtractionRequest, background_tasks: BackgroundTasks):
    """Start a metadata extraction job"""
    job_id = str(uuid.uuid4())
   
    # Cleanup old jobs before starting new one
    cleanup_old_jobs()
    
    jobs_storage[job_id] = {
        "status": "pending",
        "progress": 0,
        "created_at": datetime.now(),
        "request": request.model_dump()
    }
   
    background_tasks.add_task(process_metadata_extraction, job_id, request)
   
    return MetadataExtractionResponse(
        jobId=job_id,
        status="pending",
        progress=0
    )

@router.get("/job/{job_id}", response_model=MetadataExtractionResponse)
async def get_job_status(job_id: str):
    """Get the status of a metadata extraction job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
   
    job = jobs_storage[job_id]
   
    return MetadataExtractionResponse(
        jobId=job_id,
        status=job["status"],
        progress=job["progress"],
        results=job.get("results"),
        error=job.get("error"),
        processing_stats=job.get("processing_stats"),
        feedback=job.get("feedback")
    )

@router.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download the results as a CSV file - Memory optimized"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
   
    job = jobs_storage[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
   
    results = job.get("results", [])
    if not results:
        raise HTTPException(status_code=404, detail="No results found")
   
    try:
        # Convert results to CSV with validation - Memory efficient approach
        csv_data = []
        missing_metadata_count = 0
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for batch_start in range(0, len(results), batch_size):
            batch_end = min(batch_start + batch_size, len(results))
            batch_results = results[batch_start:batch_end]
            
            for result in batch_results:
                row = {
                    "Row Index": result.rowIndex,
                    "Original Text": result.originalText[:1000] if len(result.originalText) > 1000 else result.originalText  # Truncate long text
                }
           
                for field_name, field_result in result.extractedMetadata.items():
                    row[f"{field_name}"] = field_result.value
                    row[f"{field_name} Confidence"] = field_result.confidence
                    row[f"{field_name} Reason"] = field_result.reason[:200] if len(field_result.reason) > 200 else field_result.reason  # Truncate long reasons
           
                expected_fields = job.get("request", {}).get("config", {}).get("metadataFields", [])
                for expected_field in expected_fields:
                    field_name = expected_field.get("name", "")
                    if f"{field_name}" not in row:
                        row[f"{field_name}"] = "Missing"
                        row[f"{field_name} Confidence"] = "Low"
                        row[f"{field_name} Reason"] = "Field was missing from extraction results"
                        missing_metadata_count += 1
           
                csv_data.append(row)
            
            # Force garbage collection after each batch
            gc.collect()
   
        print(f"CSV Generation Summary:")
        print(f"Total rows in CSV: {len(csv_data)}")
        print(f"Missing metadata fields: {missing_metadata_count}")
   
        if csv_data:
            print(f"CSV columns: {len(list(csv_data[0].keys()))}")
           
            metadata_fields = [key for key in csv_data[0].keys() if key not in ["Row Index", "Original Text"]]
            if metadata_fields:
                print(f"CSV contains {len(metadata_fields)} metadata-related columns")
            else:
                print("WARNING: CSV only contains Row Index and Original Text columns")
           
            df = pd.DataFrame(csv_data)
            csv_string = df.to_csv(index=False)
           
            # Extract report name for filename
            report_name = "metadata_extraction"  # Default
            try:
                job_request = job.get("request", {})
                if job_request and "reportName" in job_request:
                    report_name = job_request["reportName"]
                elif job_request and "config" in job_request:
                    config = job_request["config"]
                    if "reportName" in config:
                        report_name = config["reportName"]
                
                # Clean filename
                report_name = re.sub(r'[<>:"/\\|?*]', '_', report_name)
                report_name = report_name.strip()
                if not report_name:
                    report_name = "metadata_extraction"
                    
            except Exception as e:
                print(f"Error extracting report name: {e}")
                report_name = "metadata_extraction"
           
            # Clear memory
            del csv_data
            del df
            gc.collect()
            
            from fastapi.responses import Response
            return Response(
                content=csv_string,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={report_name}_{job_id}.csv"}
            )
        else:
            raise HTTPException(status_code=404, detail="No data to download")
            
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate CSV: {str(e)}")

@router.post("/reprocess/{job_id}", response_model=MetadataExtractionResponse)
async def reprocess_rows(job_id: str, request: ReprocessRequest, background_tasks: BackgroundTasks):
    """Reprocess specific rows of a completed job"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
   
    job = jobs_storage[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Can only reprocess completed jobs")
   
    new_job_id = str(uuid.uuid4())
   
    original_request = job["request"]
    filtered_data = []
    for row_index in request.rowIndices:
        if row_index < len(original_request["csvData"]):
            filtered_data.append(original_request["csvData"][row_index])
   
    new_request = MetadataExtractionRequest(
        config=MetadataExtractionConfig(**original_request["config"]),
        csvData=filtered_data
    )
   
    jobs_storage[new_job_id] = {
        "status": "pending",
        "progress": 0,
        "created_at": datetime.now(),
        "request": new_request.model_dump(),
        "parent_job": job_id
    }
   
    background_tasks.add_task(process_metadata_extraction, new_job_id, new_request)
   
    return MetadataExtractionResponse(
        jobId=new_job_id,
        status="pending",
        progress=0
    )

@router.get("/debug/{job_id}")
async def debug_job_results(job_id: str):
    """Debug endpoint to check the structure of job results"""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
   
    job = jobs_storage[job_id]
    results = job.get("results", [])
   
    if not results:
        return {"detail": "No results found"}
   
    sample_result = results[0]
   
    structure_info = {
        "result_type": str(type(sample_result)),
        "has_rowIndex": hasattr(sample_result, "rowIndex"),
        "has_originalText": hasattr(sample_result, "originalText"),
        "has_extractedMetadata": hasattr(sample_result, "extractedMetadata"),
    }
   
    if hasattr(sample_result, "extractedMetadata"):
        metadata = sample_result.extractedMetadata
        first_key = next(iter(metadata.keys()), None)
       
        if first_key:
            first_field = metadata[first_key]
            structure_info["metadata_field_type"] = str(type(first_field))
            structure_info["field_has_value"] = hasattr(first_field, "value")
            structure_info["field_has_confidence"] = hasattr(first_field, "confidence")
            structure_info["field_has_reason"] = hasattr(first_field, "reason")
   
    return {
        "structure_info": structure_info,
        "sample_keys": dir(sample_result) if hasattr(sample_result, "__dict__") else [],
        "first_few_results": [r.model_dump() if hasattr(r, "model_dump") else r for r in results[:2]]
    }