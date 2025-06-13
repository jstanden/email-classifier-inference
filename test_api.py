#!/usr/bin/env python3
"""
Test script for the Email Classifier Inference API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_classification():
    """Test the classification endpoint"""
    print("\nTesting classification endpoint...")
    
    test_emails = [
        {
            "subject": "Invoice for services",
            "body": "Please find attached the invoice for the services provided last month. Total amount due: $500."
        },
        {
            "subject": "Technical support request",
            "body": "I'm having trouble logging into my account. Can you help me reset my password?"
        },
        {
            "subject": "Special offer - 50% off!",
            "body": "Don't miss our limited time offer! Get 50% off all premium features this week only."
        },
        {
            "subject": "API Integration Question",
            "body": "I'm trying to integrate your API with our system. Do you have any documentation for the webhook endpoints?"
        },
        {
            "subject": "Windows Installation Issue",
            "body": "I'm getting an error when trying to install the software on Windows 10. The installer keeps failing."
        },
        {
            "subject": "iOS App Update",
            "body": "When will the new version of the iOS app be available in the App Store? I'm waiting for the new features."
        },
        {
            "subject": "Community Forum Access",
            "body": "I'd like to join the community forum to connect with other users and share best practices."
        },
        {
            "subject": "Security Certificate Renewal",
            "body": "Our security certificates are expiring soon. How do we renew them for the CX Security module?"
        }
    ]
    
    for i, email in enumerate(test_emails, 1):
        print(f"\nTest email {i}:")
        print(f"Subject: {email['subject']}")
        print(f"Body: {email['body']}")
        
        # Test default mode (top prediction only)
        print("  Testing default mode (top prediction only):")
        try:
            response = requests.post(
                f"{BASE_URL}/classify",
                json=email,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Top classification: {result['classifications'][0]['label']} ({result['classifications'][0]['score']:.4f})")
                print(f"  Total predictions returned: {len(result['classifications'])}")
            else:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"  Error: {e}")
        
        # Test show_all_scores mode
        print("  Testing show_all_scores mode:")
        try:
            response = requests.post(
                f"{BASE_URL}/classify?show_all_scores=true",
                json=email,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"  Top 3 classifications:")
                for j, classification in enumerate(result['classifications'][:3], 1):
                    print(f"    {j}. {classification['label']}: {classification['score']:.4f}")
                print(f"  Total predictions returned: {len(result['classifications'])}")
            else:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            print(f"  Error: {e}")

def test_batch_classification():
    """Test the batch classification endpoint"""
    print("\nTesting batch classification endpoint...")
    
    batch_emails = [
        {
            "subject": "Invoice for services",
            "body": "Please find attached the invoice for the services provided last month. Total amount due: $500."
        },
        {
            "subject": "Technical support request",
            "body": "I'm having trouble logging into my account. Can you help me reset my password?"
        },
        {
            "subject": "Special offer - 50% off!",
            "body": "Don't miss our limited time offer! Get 50% off all premium features this week only."
        },
        {
            "subject": "API Integration Question",
            "body": "I'm trying to integrate your API with our system. Do you have any documentation for the webhook endpoints?"
        },
        {
            "subject": "Windows Installation Issue",
            "body": "I'm getting an error when trying to install the software on Windows 10. The installer keeps failing."
        }
    ]
    
    batch_request = {"emails": batch_emails}
    
    print(f"Testing batch classification with {len(batch_emails)} emails...")
    
    # Test default mode (top prediction only)
    print("\nTesting default mode (top prediction only):")
    try:
        response = requests.post(
            f"{BASE_URL}/classify-batch",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Batch processing completed:")
            print(f"  Total emails: {result['total_emails']}")
            print(f"  Processing time: {result['processing_time_ms']}ms")
            print(f"  Average time per email: {result['processing_time_ms'] / result['total_emails']:.2f}ms")
            
            print("\nResults (top prediction only):")
            for i, email_result in enumerate(result['results'], 1):
                print(f"  Email {i}: {email_result['classifications'][0]['label']} ({email_result['classifications'][0]['score']:.4f})")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    # Test show_all_scores mode
    print("\nTesting show_all_scores mode:")
    try:
        response = requests.post(
            f"{BASE_URL}/classify-batch?show_all_scores=true",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Batch processing completed:")
            print(f"  Total emails: {result['total_emails']}")
            print(f"  Processing time: {result['processing_time_ms']}ms")
            print(f"  Average time per email: {result['processing_time_ms'] / result['total_emails']:.2f}ms")
            
            print("\nResults (all scores):")
            for i, email_result in enumerate(result['results'], 1):
                print(f"  Email {i} top 3 classifications:")
                for j, classification in enumerate(email_result['classifications'][:3], 1):
                    print(f"    {j}. {classification['label']}: {classification['score']:.4f}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

def test_batch_validation():
    """Test batch endpoint validation"""
    print("\nTesting batch validation...")
    
    # Test empty batch
    print("Testing empty batch...")
    try:
        response = requests.post(
            f"{BASE_URL}/classify-batch",
            json={"emails": []},
            headers={"Content-Type": "application/json"}
        )
        print(f"Empty batch status: {response.status_code}")
        if response.status_code == 400:
            print("✅ Correctly rejected empty batch")
        else:
            print("❌ Should have rejected empty batch")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test oversized batch (would need 101 emails, but we'll test with a smaller limit)
    print("\nTesting batch size limit...")
    large_batch = {"emails": [{"subject": f"Test {i}", "body": f"Test body {i}"} for i in range(5)]}
    try:
        response = requests.post(
            f"{BASE_URL}/classify-batch",
            json=large_batch,
            headers={"Content-Type": "application/json"}
        )
        print(f"Large batch status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Large batch processed successfully")
        else:
            print(f"❌ Large batch failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Run all tests"""
    print("Email Classifier API Test Suite")
    print("=" * 40)
    
    # Wait a moment for the server to be ready
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    # Test health endpoint
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Make sure the server is running.")
        return
    
    # Test model info
    model_ok = test_model_info()
    
    if not model_ok:
        print("\n❌ Model info failed. Check if the model loaded correctly.")
        return
    
    # Test single classification
    test_classification()
    
    # Test batch classification
    test_batch_classification()
    
    # Test batch validation
    test_batch_validation()
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main() 