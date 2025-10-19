"""
Simple FastAPI Backend for Job Market Analytics
===============================================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI(
    title="Job Market Analytics API",
    description="Simple REST API for job market analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Job Market Analytics API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint."""
    return {"message": "API is working!", "data": [1, 2, 3, 4, 5]}
