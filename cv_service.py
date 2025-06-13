#!/usr/bin/env python3
"""
CV Hacker Service - FastAPI Web Service for CV Optimization
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import shutil
import asyncio
import uuid
import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
from dotenv import load_dotenv

# Import our CV processing functions
from cv_processor import CVProcessor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="CV Hacker Service",
    description="AI-powered CV optimization service using Claude 4",
    version="1.0.0"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# In-memory task storage (use Redis/database in production)
task_storage: Dict[str, Dict[str, Any]] = {}

# Models
class OptimizationRequest(BaseModel):
    job_description: str
    prompt_type: str = "normal"  # "normal" or "unhinged"

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class OptimizationResult(BaseModel):
    task_id: str
    status: str
    analysis: Optional[str] = None
    optimized_cv_latex: Optional[str] = None
    cover_letter_latex: Optional[str] = None
    files_generated: Optional[Dict[str, str]] = None
    processing_time: Optional[float] = None

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token or API key."""
    token = credentials.credentials
    
    try:
        # Option 1: JWT token verification
        if token.startswith('eyJ'):  # JWT tokens start with eyJ
            secret_key = os.getenv("JWT_SECRET_KEY")
            if not secret_key:
                raise HTTPException(status_code=401, detail="JWT secret not configured")
            
            payload = jwt.decode(token, secret_key, algorithms=["HS256"])
            user_id = payload.get("userId")  # Changed from "user_id" to "userId" to match backend
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token payload")
            
            return {"user_id": user_id, "auth_type": "jwt"}
        
        # Option 2: API key verification
        else:
            valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
            if token not in valid_api_keys:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            return {"user_id": f"api_key_user_{hash(token) % 10000}", "auth_type": "api_key"}
    
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# Helper functions
def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file."""
    allowed_extensions = {".pdf", ".tex"}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return False
    
    # Check file size (max 10MB)
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        return False
    
    return True

def validate_prompt_type(prompt_type: str) -> bool:
    """Validate prompt type."""
    return prompt_type.lower() in ["normal", "unhinged"]

async def process_cv_optimization(
    task_id: str,
    file_path: str,
    job_description: str,
    prompt_type: str,
    user_info: dict
):
    """Background task for CV optimization."""
    start_time = datetime.now()
    
    try:
        # Update task status
        task_storage[task_id]["status"] = "processing"
        
        # Initialize CV processor
        processor = CVProcessor()
        
        # Process the CV
        result = await processor.process_cv(
            file_path=file_path,
            job_description=job_description,
            prompt_type=prompt_type.lower(),
            user_info=user_info
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update task with results
        task_storage[task_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "result": {
                "analysis": result.get("analysis"),
                "optimized_cv_latex": result.get("cv_latex"),
                "cover_letter_latex": result.get("cover_latex"),
                "files_generated": result.get("files"),
                "processing_time": processing_time
            }
        })
        
        logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        task_storage[task_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "error": str(e)
        })
    
    finally:
        # Clean up uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/optimize", response_model=Dict[str, str])
async def optimize_cv(
    background_tasks: BackgroundTasks,
    cv_file: UploadFile = File(...),
    job_description: str = Form(...),
    prompt_type: str = Form("normal"),
    user_info: dict = Depends(verify_token)
):
    """
    Optimize CV with AI analysis.
    
    - **cv_file**: PDF or LaTeX file to optimize
    - **job_description**: Target job posting text
    - **prompt_type**: "normal" or "unhinged" optimization mode
    """
    
    # Validate inputs
    if not validate_file(cv_file):
        raise HTTPException(
            status_code=400,
            detail="Invalid file. Please upload a PDF or LaTeX (.tex) file under 10MB."
        )
    
    if not validate_prompt_type(prompt_type):
        raise HTTPException(
            status_code=400,
            detail="Invalid prompt_type. Must be 'normal' or 'unhinged'."
        )
    
    if len(job_description.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Job description must be at least 50 characters long."
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    try:
        # Save uploaded file temporarily
        temp_dir = Path(tempfile.gettempdir()) / "cv_hacker" / task_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_extension = Path(cv_file.filename).suffix
        temp_file_path = temp_dir / f"cv{file_extension}"
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(cv_file.file, buffer)
        
        # Initialize task
        task_storage[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "created_at": datetime.now(),
            "user_id": user_info["user_id"],
            "prompt_type": prompt_type,
            "filename": cv_file.filename
        }
        
        # Start background processing
        background_tasks.add_task(
            process_cv_optimization,
            task_id,
            str(temp_file_path),
            job_description,
            prompt_type,
            user_info
        )
        
        logger.info(f"Started optimization task {task_id} for user {user_info['user_id']}")
        
        return {
            "task_id": task_id,
            "status": "pending",
            "message": "CV optimization started. Use the task_id to check status."
        }
    
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to start CV optimization")

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(
    task_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get the status of a CV optimization task."""
    
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    
    # Check if user owns this task
    if task["user_id"] != user_info["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return TaskStatus(**task)

@app.get("/result/{task_id}", response_model=OptimizationResult)
async def get_optimization_result(
    task_id: str,
    user_info: dict = Depends(verify_token)
):
    """Get the complete result of a CV optimization task."""
    
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    
    # Check if user owns this task
    if task["user_id"] != user_info["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed. Current status: {task['status']}"
        )
    
    result_data = {
        "task_id": task_id,
        "status": task["status"]
    }
    
    if task.get("result"):
        result_data.update(task["result"])
    
    return OptimizationResult(**result_data)

@app.delete("/task/{task_id}")
async def delete_task(
    task_id: str,
    user_info: dict = Depends(verify_token)
):
    """Delete a task and its results."""
    
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_storage[task_id]
    
    # Check if user owns this task
    if task["user_id"] != user_info["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Clean up any remaining files
    temp_dir = Path(tempfile.gettempdir()) / "cv_hacker" / task_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Remove from storage
    del task_storage[task_id]
    
    return {"message": "Task deleted successfully"}

@app.get("/tasks")
async def list_user_tasks(
    user_info: dict = Depends(verify_token),
    limit: int = 10
):
    """List all tasks for the authenticated user."""
    
    user_tasks = [
        {
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"],
            "completed_at": task.get("completed_at"),
            "prompt_type": task.get("prompt_type"),
            "filename": task.get("filename")
        }
        for task_id, task in task_storage.items()
        if task["user_id"] == user_info["user_id"]
    ]
    
    # Sort by creation time, most recent first
    user_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {"tasks": user_tasks[:limit]}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Cleanup task (run periodically to remove old tasks)
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    logger.info("CV Hacker Service starting up...")
    
    # Create temp directories
    temp_base = Path(tempfile.gettempdir()) / "cv_hacker"
    temp_base.mkdir(exist_ok=True)
    
    # Start cleanup task
    asyncio.create_task(cleanup_old_tasks())

async def cleanup_old_tasks():
    """Clean up old tasks periodically."""
    while True:
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            tasks_to_remove = [
                task_id for task_id, task in task_storage.items()
                if task["created_at"] < cutoff_time
            ]
            
            for task_id in tasks_to_remove:
                # Clean up files
                temp_dir = Path(tempfile.gettempdir()) / "cv_hacker" / task_id
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                
                # Remove from storage
                del task_storage[task_id]
            
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        # Sleep for 1 hour
        await asyncio.sleep(3600)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "service:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"
    )