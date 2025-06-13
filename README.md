# Email Classifier Inference API

A FastAPI server for email classification using a HuggingFace transformer model (`email_classifier_final`).

## Features

- **Email Classification**: Classify emails based on subject and body text
- **Batch Processing**: Process multiple emails in a single request for better efficiency
- **Confidence Scores**: Get confidence scores for all classification labels
- **OpenAPI Documentation**: Interactive API documentation
- **Health Checks**: Monitor API and model status
- **GPU Support**: Automatic GPU detection and utilization (MPS, CUDA, CPU)
- **Docker Support**: Easy containerized deployment
- **Consistent Preprocessing**: Uses inference_hook for training-inference consistency

## Setup

### Prerequisites

- Python 3.8+ (for local development)
- Docker & Docker Compose (for containerized deployment)
- The `email_classifier_final` model files in the `models/` directory

### Installation

#### Option 1: Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model files are in the correct location:
```
models/
└── email_classifier_final/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── ...
```

3. Run the server:
```bash
python main.py
```

Or using the startup script:
```bash
./start_server.sh
```

#### Option 2: Docker Deployment (Recommended)

1. **Quick Start**:
```bash
./deploy.sh run
```

2. **Manual Docker Commands**:
```bash
# Build the image
docker build -t email-classifier-api .

# Run the container
docker run -p 8000:8000 email-classifier-api
```

3. **Using Docker Compose**:
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Deployment Script

The `deploy.sh` script provides easy management commands:

```bash
./deploy.sh build     # Build the Docker image
./deploy.sh run       # Build and start the API
./deploy.sh stop      # Stop the container
./deploy.sh restart   # Restart the container
./deploy.sh logs      # View logs
./deploy.sh status    # Check API health
./deploy.sh cleanup   # Remove all containers and images
```

The server will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
- **GET** `/health`
- Returns the status of the API and model

### 2. Model Information
- **GET** `/model-info`
- Returns information about the loaded model

### 3. Single Email Classification
- **POST** `/classify`
- Classifies a single email based on subject and body
- **Query Parameters:**
  - `show_all_scores` (bool, default: false): If true, returns all classification scores. If false, returns only the top prediction.

### 4. Batch Email Classification
- **POST** `/classify-batch`
- Classifies multiple emails in a single request (up to 100 emails)
- More efficient for processing large volumes of emails
- **Query Parameters:**
  - `show_all_scores` (bool, default: false): If true, returns all classification scores for each email. If false, returns only the top prediction for each email.

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model-info

# Classify a single email (top prediction only - default)
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "subject": "Invoice for services",
       "body": "Please find attached the invoice for the services provided last month."
     }'

# Classify a single email (all scores)
curl -X POST "http://localhost:8000/classify?show_all_scores=true" \
     -H "Content-Type: application/json" \
     -d '{
       "subject": "Invoice for services",
       "body": "Please find attached the invoice for the services provided last month."
     }'

# Classify multiple emails (top prediction only - default)
curl -X POST "http://localhost:8000/classify-batch" \
     -H "Content-Type: application/json" \
     -d '{
       "emails": [
         {
           "subject": "Invoice for services",
           "body": "Please find attached the invoice for the services provided last month."
         },
         {
           "subject": "Technical support request",
           "body": "I need help with my account login."
         },
         {
           "subject": "Special offer",
           "body": "Get 50% off all premium features this week!"
         }
       ]
     }'

# Classify multiple emails (all scores)
curl -X POST "http://localhost:8000/classify-batch?show_all_scores=true" \
     -H "Content-Type: application/json" \
     -d '{
       "emails": [
         {
           "subject": "Invoice for services",
           "body": "Please find attached the invoice for the services provided last month."
         },
         {
           "subject": "Technical support request",
           "body": "I need help with my account login."
         },
         {
           "subject": "Special offer",
           "body": "Get 50% off all premium features this week!"
         }
       ]
     }'
```

### Using Python requests

```python
import requests

# Single email classification (top prediction only - default)
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "subject": "Invoice for services",
        "body": "Please find attached the invoice for the services provided last month."
    }
)

result = response.json()
print("Top classification:")
top_class = result["classifications"][0]
print(f"  {top_class['label']}: {top_class['score']:.4f}")

# Single email classification (all scores)
response = requests.post(
    "http://localhost:8000/classify?show_all_scores=true",
    json={
        "subject": "Invoice for services",
        "body": "Please find attached the invoice for the services provided last month."
    }
)

result = response.json()
print("All classifications:")
for classification in result["classifications"]:
    print(f"  {classification['label']}: {classification['score']:.4f}")

# Batch email classification (top prediction only - default)
batch_response = requests.post(
    "http://localhost:8000/classify-batch",
    json={
        "emails": [
            {
                "subject": "Invoice for services",
                "body": "Please find attached the invoice for the services provided last month."
            },
            {
                "subject": "Technical support request",
                "body": "I need help with my account login."
            },
            {
                "subject": "Special offer",
                "body": "Get 50% off all premium features this week!"
            }
        ]
    }
)

batch_result = batch_response.json()
print(f"Batch processing completed:")
print(f"  Total emails: {batch_result['total_emails']}")
print(f"  Processing time: {batch_result['processing_time_ms']}ms")

for i, email_result in enumerate(batch_result['results']):
    print(f"\nEmail {i+1} top classification:")
    top_class = email_result['classifications'][0]
    print(f"  {top_class['label']}: {top_class['score']:.4f}")

# Batch email classification (all scores)
batch_response = requests.post(
    "http://localhost:8000/classify-batch?show_all_scores=true",
    json={
        "emails": [
            {
                "subject": "Invoice for services",
                "body": "Please find attached the invoice for the services provided last month."
            },
            {
                "subject": "Technical support request",
                "body": "I need help with my account login."
            },
            {
                "subject": "Special offer",
                "body": "Get 50% off all premium features this week!"
            }
        ]
    }
)

batch_result = batch_response.json()
print(f"Batch processing completed:")
print(f"  Total emails: {batch_result['total_emails']}")
print(f"  Processing time: {batch_result['processing_time_ms']}ms")

for i, email_result in enumerate(batch_result['results']):
    print(f"\nEmail {i+1} all classifications:")
    for j, classification in enumerate(email_result['classifications'], 1):
        print(f"  {j}. {classification['label']}: {classification['score']:.4f}")
```

### Example Responses

#### Single Email Response (Default - Top Prediction Only)
```json
{
  "classifications": [
    {
      "label": "Billing",
      "score": 0.9234
    }
  ]
}
```

#### Single Email Response (All Scores)
```json
{
  "classifications": [
    {
      "label": "Billing",
      "score": 0.9234
    },
    {
      "label": "Support",
      "score": 0.0456
    },
    {
      "label": "Marketing",
      "score": 0.0310
    }
  ]
}
```

#### Batch Email Response (Default - Top Prediction Only)
```json
{
  "results": [
    {
      "classifications": [
        {
          "label": "Billing",
          "score": 0.9234
        }
      ]
    },
    {
      "classifications": [
        {
          "label": "Support",
          "score": 0.8567
        }
      ]
    }
  ],
  "total_emails": 2,
  "processing_time_ms": 245.67
}
```

#### Batch Email Response (All Scores)
```json
{
  "results": [
    {
      "classifications": [
        {
          "label": "Billing",
          "score": 0.9234
        },
        {
          "label": "Support",
          "score": 0.0456
        },
        {
          "label": "Marketing",
          "score": 0.0310
        }
      ]
    },
    {
      "classifications": [
        {
          "label": "Support",
          "score": 0.8567
        },
        {
          "label": "Billing",
          "score": 0.1234
        },
        {
          "label": "Marketing",
          "score": 0.0199
        }
      ]
    }
  ],
  "total_emails": 2,
  "processing_time_ms": 245.67
}
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Model Details

The API uses the `email_classifier_final` model, which is a text classification model trained to categorize emails. The model:

- Combines email subject and body for classification
- Returns confidence scores for all possible labels
- Supports GPU acceleration when available (MPS, CUDA, CPU)
- Automatically loads on server startup
- Supports 15 classification labels: Memberships, Integrations, CX Security, B5, Marketing, CX Community, Extensions, Windows, XAM, Linux, Mac, MSP, Billing, Android, iOS
- Uses consistent preprocessing via `inference_hook.py`

## Performance Considerations

### Single vs Batch Processing

- **Single Email (`/classify`)**: Best for real-time classification of individual emails
- **Batch Email (`/classify-batch`)**: More efficient for processing multiple emails, reduces HTTP overhead

### Batch Processing Limits

- Maximum 100 emails per batch request
- Processing time scales linearly with batch size
- Automatic validation prevents oversized requests

### Hardware Acceleration

The API automatically detects and uses the best available hardware:
- **MPS (Apple Silicon)**: ~3-5x faster than CPU on Mac M1/M2/M3
- **CUDA (NVIDIA)**: ~5-10x faster than CPU on systems with NVIDIA GPUs
- **CPU**: Reliable fallback for all systems

## Docker Configuration

### Container Features

- **Multi-stage build**: Optimized for size and security
- **Non-root user**: Enhanced security
- **Health checks**: Automatic monitoring
- **Resource limits**: Memory and CPU constraints
- **Volume mounts**: Optional model and log persistence

### Environment Variables

```bash
PYTHONUNBUFFERED=1    # Real-time logging
PORT=8000             # API port
```

### Resource Requirements

- **Memory**: 2-4GB recommended (model loading + inference)
- **CPU**: 2+ cores recommended
- **Storage**: ~500MB for base image + model files

## Error Handling

The API includes comprehensive error handling for:
- Model loading failures
- Classification errors
- Invalid input data
- Missing model files
- Batch size limits
- Empty batch requests
- Hardware detection issues

## Development

### Project Structure
```
.
├── main.py              # FastAPI application
├── inference_hook.py    # Preprocessing pipeline
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-container setup
├── deploy.sh           # Deployment script
├── .dockerignore       # Docker build optimization
├── README.md           # This file
├── test_api.py         # Test script
├── start_server.sh     # Local startup script
└── models/
    └── email_classifier_final/
        └── ...         # Model files
```

### Testing

Run the comprehensive test suite:
```bash
python test_api.py
```

The test suite includes:
- Health check validation
- Model info verification
- Single email classification
- Batch email classification
- Input validation testing

### Adding New Endpoints

To add new endpoints, modify `main.py` and add new route handlers following the existing pattern.

### Environment Variables

You can configure the server using environment variables:
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model files are in the correct directory
2. **CUDA out of memory**: The model will automatically fall back to CPU if GPU memory is insufficient
3. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
4. **Batch size exceeded**: Reduce the number of emails in your batch request (max 100)
5. **Docker build fails**: Check that all required files are present and not excluded by `.dockerignore`

### Docker-Specific Issues

1. **Container won't start**: Check logs with `docker-compose logs -f`
2. **Out of memory**: Increase Docker memory limits or reduce batch sizes
3. **Port conflicts**: Change the port mapping in `docker-compose.yml`
4. **Model loading slow**: Consider using volume mounts for persistent model storage

### Logs

The application logs important events including:
- Model loading status
- Classification requests
- Batch processing statistics
- Error messages
- Hardware detection results

Check the console output for detailed logs:
- **Local**: Direct console output
- **Docker**: `docker-compose logs -f`

## Production Deployment

### Recommended Setup

1. **Use Docker Compose** for easy management
2. **Set resource limits** to prevent memory issues
3. **Enable health checks** for monitoring
4. **Use volume mounts** for model persistence
5. **Configure logging** for production monitoring

### Scaling Considerations

- **Horizontal scaling**: Run multiple container instances behind a load balancer
- **Model caching**: Consider using shared model storage
- **Batch optimization**: Adjust batch sizes based on available resources
- **Monitoring**: Implement proper logging and metrics collection 