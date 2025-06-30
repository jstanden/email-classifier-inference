from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import List, Dict, Any, Optional
import os
from contextlib import asynccontextmanager
from inference_hook import customize_email_for_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_PATH = os.getenv("ROOT_PATH", "")
API_SECRET_TOKEN = os.getenv("API_SECRET_TOKEN", "")

# Global variable to store the pipeline
classifier = None

# Security scheme for bearer token
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Verify the bearer token if API_SECRET_TOKEN is set.
    If API_SECRET_TOKEN is not set or empty, no authentication is required.
    """
    # If no API secret token is configured, skip authentication
    if not API_SECRET_TOKEN:
        return None
    
    # If API secret token is configured, require authentication
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != API_SECRET_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials

def load_model():
    """Load the email classifier model"""
    global classifier
    try:
        model_path = "models/email_classifier_final"
        
        # Check if model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        
        # Determine the best available device
        device = -1  # Default to CPU
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = 0
            logger.info("Using CUDA (NVIDIA GPU)")
        else:
            logger.info("Using CPU")
        
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create the pipeline
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,  # Return all scores instead of deprecated return_all_scores=True
            device=device
        )
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup
    load_model()
    yield
    # Shutdown (if needed)
    logger.info("Shutting down Email Classifier API")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Email Classifier Inference API",
    description="API for classifying emails using HuggingFace transformers",
    version="1.0.0",
    lifespan=lifespan,
    root_path=ROOT_PATH
)

class EmailRequest(BaseModel):
    subject: str
    body: str

class ClassificationResult(BaseModel):
    label: str
    score: float

class EmailResponse(BaseModel):
    classifications: List[ClassificationResult]

class BatchEmailRequest(BaseModel):
    emails: List[EmailRequest]

class BatchEmailResponse(BaseModel):
    results: List[EmailResponse]
    total_emails: int
    processing_time_ms: float

@app.get("/")
async def root(token: Optional[str] = Depends(verify_token)):
    """Health check endpoint"""
    return {"message": "Email Classifier Inference API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Determine the actual device being used
    device_info = "cpu"
    if classifier is not None:
        try:
            model_device = str(next(classifier.model.parameters()).device)
            if "mps" in model_device:
                device_info = "mps"
            elif "cuda" in model_device:
                device_info = "cuda"
            else:
                device_info = "cpu"
        except:
            device_info = "cpu"
    
    return {
        "status": "healthy",
        "model_loaded": classifier is not None,
        "device": device_info,
        "device_available": {
            "mps": torch.backends.mps.is_available(),
            "cuda": torch.cuda.is_available(),
            "cpu": True
        }
    }

@app.post("/classify", response_model=EmailResponse)
async def classify_email(
    request: EmailRequest,
    show_all_scores: bool = Query(
        default=False, 
        description="If True, return all classification scores. If False, return only the top prediction."
    ),
    token: Optional[str] = Depends(verify_token)
):
    """
    Classify an email based on its subject and body.
    
    Args:
        request: Email data containing subject and body
        show_all_scores: If True, return all classification scores. If False, return only the top prediction.
    
    Returns:
        Classification results with confidence scores.
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Apply inference hook preprocessing for consistency with training
        email_dict = {"subject": request.subject, "body": request.body}
        processed_email = customize_email_for_inference(email_dict)
        
        # Combine subject and body for classification
        combined_text = f"{processed_email['subject']} {processed_email['body']}"
        
        # Truncate text to fit model's maximum sequence length (usually 512 tokens)
        # We'll use the tokenizer to ensure proper truncation
        tokenizer = classifier.tokenizer
        encoded = tokenizer.encode_plus(
            combined_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        truncated_text = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
        
        # Perform classification
        results = classifier(truncated_text)
        
        # Extract classifications from the first (and only) result
        all_classifications = []
        for result in results[0]:
            all_classifications.append(ClassificationResult(
                label=result['label'],
                score=float(result['score'])
            ))
        
        # Sort by confidence score (highest first)
        all_classifications.sort(key=lambda x: x.score, reverse=True)
        
        # Return all scores or just the top one based on parameter
        if show_all_scores:
            classifications = all_classifications
        else:
            classifications = [all_classifications[0]] if all_classifications else []
        
        return EmailResponse(
            classifications=classifications
        )
        
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify-batch", response_model=BatchEmailResponse)
async def classify_emails_batch(
    request: BatchEmailRequest,
    show_all_scores: bool = Query(
        default=False, 
        description="If True, return all classification scores for each email. If False, return only the top prediction for each email."
    ),
    token: Optional[str] = Depends(verify_token)
):
    """
    Classify multiple emails in a single request.
    
    This endpoint is more efficient for processing large volumes of emails
    as it reduces the overhead of multiple HTTP requests.
    
    Args:
        request: Batch of emails to classify
        show_all_scores: If True, return all classification scores for each email. If False, return only the top prediction for each email.
    
    Returns:
        Classifications for all emails with processing time information.
    """
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.emails:
        raise HTTPException(status_code=400, detail="No emails provided")
    
    if len(request.emails) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 emails per batch request")
    
    import time
    start_time = time.time()
    
    try:
        results = []
        
        for email in request.emails:
            # Apply inference hook preprocessing for consistency with training
            email_dict = {"subject": email.subject, "body": email.body}
            processed_email = customize_email_for_inference(email_dict)
            
            # Combine subject and body for classification
            combined_text = f"{processed_email['subject']} {processed_email['body']}"
            
            # Truncate text to fit model's maximum sequence length (usually 512 tokens)
            # We'll use the tokenizer to ensure proper truncation
            tokenizer = classifier.tokenizer
            encoded = tokenizer.encode_plus(
                combined_text,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            truncated_text = tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
            
            # Perform classification
            classification_results = classifier(truncated_text)
            
            # Extract classifications from the first (and only) result
            all_classifications = []
            for result in classification_results[0]:
                all_classifications.append(ClassificationResult(
                    label=result['label'],
                    score=float(result['score'])
                ))
            
            # Sort by confidence score (highest first)
            all_classifications.sort(key=lambda x: x.score, reverse=True)
            
            # Return all scores or just the top one based on parameter
            if show_all_scores:
                classifications = all_classifications
            else:
                classifications = [all_classifications[0]] if all_classifications else []
            
            results.append(EmailResponse(
                classifications=classifications
            ))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info(f"Batch classification completed: {len(request.emails)} emails in {processing_time:.2f}ms")
        
        return BatchEmailResponse(
            results=results,
            total_emails=len(request.emails),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error during batch classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

@app.get("/model-info")
async def get_model_info(token: Optional[str] = Depends(verify_token)):
    """Get information about the loaded model"""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get model configuration
        model = classifier.model
        config = model.config
        
        return {
            "model_name": "email_classifier_final",
            "model_type": config.model_type,
            "num_labels": config.num_labels,
            "labels": config.label2id,
            "device": str(next(model.parameters()).device),
            "model_path": "models/email_classifier_final"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 