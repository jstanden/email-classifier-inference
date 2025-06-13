#!/bin/bash

# Email Classifier API Docker Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Check if model files exist
check_models() {
    if [ ! -d "models/email_classifier_final" ]; then
        print_error "Model directory not found at models/email_classifier_final"
        print_error "Please ensure the model files are in the correct location"
        exit 1
    fi
    
    if [ ! -f "models/email_classifier_final/model.safetensors" ]; then
        print_warning "Model file not found. The container may fail to start."
    fi
}

# Build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t email-classifier-api .
    print_success "Docker image built successfully"
}

# Run the container
run_container() {
    print_status "Starting email classifier API container..."
    docker-compose up -d
    print_success "Container started successfully"
    
    # Wait for the service to be ready
    print_status "Waiting for service to be ready..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy and ready!"
        print_status "API documentation available at: http://localhost:8000/docs"
    else
        print_warning "API may still be starting up. Check logs with: docker-compose logs -f"
    fi
}

# Stop the container
stop_container() {
    print_status "Stopping email classifier API container..."
    docker-compose down
    print_success "Container stopped successfully"
}

# Show logs
show_logs() {
    print_status "Showing container logs..."
    docker-compose logs -f
}

# Show status
show_status() {
    print_status "Container status:"
    docker-compose ps
    
    print_status "Health check:"
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "API is healthy"
        curl -s http://localhost:8000/health | python3 -m json.tool
    else
        print_error "API is not responding"
    fi
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes --remove-orphans
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-help}" in
    "build")
        check_docker
        check_models
        build_image
        ;;
    "run"|"start")
        check_docker
        check_models
        build_image
        run_container
        ;;
    "stop")
        check_docker
        stop_container
        ;;
    "restart")
        check_docker
        stop_container
        sleep 2
        run_container
        ;;
    "logs")
        check_docker
        show_logs
        ;;
    "status")
        check_docker
        show_status
        ;;
    "cleanup")
        check_docker
        cleanup
        ;;
    "help"|*)
        echo "Email Classifier API Docker Deployment Script"
        echo "=============================================="
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  run       - Build and start the container"
        echo "  start     - Same as 'run'"
        echo "  stop      - Stop the container"
        echo "  restart   - Restart the container"
        echo "  logs      - Show container logs"
        echo "  status    - Show container status and health"
        echo "  cleanup   - Stop and remove all containers, images, and volumes"
        echo "  help      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 build    # Build the image"
        echo "  $0 run      # Build and start the API"
        echo "  $0 logs     # View logs"
        echo "  $0 status   # Check if API is running"
        ;;
esac 