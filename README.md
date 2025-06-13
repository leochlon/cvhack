# CV Service - AI-Powered Resume Optimization

This is the CV processing service for the AI Careers platform, providing intelligent resume analysis and optimization capabilities.

## Features

- **AI-Powered CV Analysis**: Advanced analysis of resume content, structure, and formatting
- **Job-Specific Optimization**: Tailored recommendations based on job descriptions
- **ATS Compatibility**: Ensures resumes are optimized for Applicant Tracking Systems
- **Skills Gap Analysis**: Identifies missing skills and suggests improvements
- **Professional Formatting**: Recommends structure and layout improvements

## Technology Stack

- **Python 3.11+**
- **FastAPI**: High-performance async web framework
- **OpenAI GPT**: Advanced language model for CV analysis
- **Docker**: Containerized deployment
- **Fly.io**: Cloud deployment platform

## API Endpoints

### POST `/process-cv`
Processes and optimizes a CV based on the provided job description.

**Request Body:**
```json
{
  "cv_text": "string",
  "job_description": "string",
  "user_id": "string"
}
```

**Response:**
```json
{
  "task_id": "string",
  "status": "processing"
}
```

### GET `/result/{task_id}`
Retrieves the processing result for a given task.

**Response:**
```json
{
  "status": "completed",
  "result": {
    "optimized_cv": "string",
    "suggestions": ["string"],
    "score": "number"
  }
}
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/leochlon/cvhack.git
cd cvhack
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
export OPENAI_API_KEY=your_openai_api_key
export PORT=8000
```

5. Run the service:
```bash
python cv_service.py
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t cv-service .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key cv-service
```

### Fly.io Deployment

1. Install Fly CLI and login:
```bash
flyctl auth login
```

2. Deploy the service:
```bash
flyctl deploy
```

## Configuration

The service can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PORT`: Port to run the service on (default: 8000)
- `HOST`: Host to bind to (default: 0.0.0.0)

## CV Analysis Capabilities

The service provides comprehensive CV analysis including:

1. **Content Analysis**:
   - Skills extraction and matching
   - Experience relevance assessment
   - Achievement quantification review

2. **Structure Optimization**:
   - Section organization recommendations
   - Length optimization suggestions
   - Formatting improvements

3. **ATS Optimization**:
   - Keyword optimization
   - Format compatibility checks
   - Parsing-friendly structure

4. **Job Matching**:
   - Job description alignment
   - Missing requirements identification
   - Tailored recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the AI Careers platform.

## Support

For support and questions, please contact the development team.
