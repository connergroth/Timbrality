import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:postgronner34@localhost:5432/Sonance')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# AOTY API configuration
AOTY_API_URL = os.getenv('AOTY_API_URL', 'http://localhost:8000')

# Redis configuration
REDIS_URL = os.getenv('UPSTASH_REDIS_REST_URL')
REDIS_TOKEN = os.getenv('UPSTASH_REDIS_REST_TOKEN')