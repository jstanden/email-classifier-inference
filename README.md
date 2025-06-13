# Email Classifier Inference API

A FastAPI server for email classification using a HuggingFace transformer model (`email_classifier_final`).

## Features

- **Email Classification**: Classify emails based on subject and body text
- **Batch Processing**: Process multiple emails in a single request for better efficiency
- **Confidence Scores**: Get confidence scores for all classification labels
- **OpenAPI Documentation**: Interactive API documentation
- **Health Checks**: Monitor API and model status
- **GPU Support**: Automatic GPU detection and utilization

## Setup

### Prerequisites

- Python 3.8+
- The `email_classifier_final` model files in the `models/` directory

### Installation

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

### Running the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
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

### 4. Batch Email Classification
- **POST** `/classify-batch`
- Classifies multiple emails in a single request (up to 100 emails)
- More efficient for processing large volumes of emails

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model-info

# Classify a single email
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{
       "subject": "Invoice for services",
       "body": "Please find attached the invoice for the services provided last month."
     }'

# Classify multiple emails
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
```

### Using Python requests

```python
import requests

# Single email classification
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "subject": "Invoice for services",
        "body": "Please find attached the invoice for the services provided last month."
    }
)

result = response.json()
print("Classifications:")
for classification in result["classifications"]:
    print(f"  {classification['label']}: {classification['score']:.4f}")

# Batch email classification
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
```

### Example Responses

#### Single Email Response
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
  ],
  "combined_text": "Subject: Invoice for services\n\nBody: Please find attached the invoice for the services provided last month."
}
```

#### Batch Email Response
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
        }
      ],
      "combined_text": "Subject: Invoice for services\n\nBody: Please find attached the invoice for the services provided last month."
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
        }
      ],
      "combined_text": "Subject: Technical support request\n\nBody: I need help with my account login."
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
- Supports GPU acceleration when available
- Automatically loads on server startup
- Supports 15 classification labels: Memberships, Integrations, CX Security, B5, Marketing, CX Community, Extensions, Windows, XAM, Linux, Mac, MSP, Billing, Android, iOS

## Performance Considerations

### Single vs Batch Processing

- **Single Email (`/classify`)**: Best for real-time classification of individual emails
- **Batch Email (`/classify-batch`)**: More efficient for processing multiple emails, reduces HTTP overhead

### Batch Processing Limits

- Maximum 100 emails per batch request
- Processing time scales linearly with batch size
- Automatic validation prevents oversized requests

## Error Handling

The API includes comprehensive error handling for:
- Model loading failures
- Classification errors
- Invalid input data
- Missing model files
- Batch size limits
- Empty batch requests

## Development

### Project Structure
```
.
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── test_api.py         # Test script
├── start_server.sh     # Startup script
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

### Logs

The application logs important events including:
- Model loading status
- Classification requests
- Batch processing statistics
- Error messages

Check the console output for detailed logs. 