#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""
import os
import sys

def test_imports():
    """Test all the imports needed for the enhanced collaborative filtering system"""
    print("Testing imports for Enhanced Collaborative Filtering System...")
    
    try:
        print("[OK] Basic Python imports...")
        import asyncio
        import logging
        import numpy as np
        import pandas as pd
        from typing import List, Dict, Any, Optional
        from datetime import datetime, timedelta
        from collections import defaultdict
        import math
        
        print("[OK] SQLAlchemy imports...")
        from sqlalchemy import text, and_, func
        from sqlalchemy.orm import Session
        
        print("[OK] Scipy imports...")
        from scipy.spatial.distance import cosine
        from scipy.stats import entropy
        
        print("[OK] Database models...")
        from models.database import get_db
        from models.collaborative_filtering import (
            LastfmUser, UserTrackInteraction, UserAlbumInteraction, 
            UserArtistInteraction, UserSimilarity, CollaborativeRecommendation
        )
        
        print("[OK] Service imports...")
        from services.lastfm_service import LastFMService
        from services.collaborative_filtering_service import CollaborativeFilteringService
        from services.enhanced_collaborative_filtering import EnhancedCollaborativeFilteringService
        
        print("[OK] Utility imports...")
        from utils.similarity_calculator import SimilarityCalculator
        
        print("\n[SUCCESS] All imports successful!")
        
        # Test basic service initialization
        print("\n[TEST] Testing service initialization...")
        
        lastfm_service = LastFMService()
        print("[OK] LastFMService initialized")
        
        cf_service = CollaborativeFilteringService()
        print("[OK] CollaborativeFilteringService initialized")
        
        enhanced_cf_service = EnhancedCollaborativeFilteringService()
        print("[OK] EnhancedCollaborativeFilteringService initialized")
        
        similarity_calculator = SimilarityCalculator()
        print("[OK] SimilarityCalculator initialized")
        
        print("\n[SUCCESS] All services initialized successfully!")
        
        # Check environment variables
        print("\n[CONFIG] Environment variable check...")
        api_key = os.getenv("API_KEY") or os.getenv("LASTFM_API_KEY")
        if api_key:
            print(f"[OK] Last.fm API key found (length: {len(api_key)})")
        else:
            print("[WARN]  Last.fm API key not found. Set API_KEY or LASTFM_API_KEY environment variable.")
            
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            print("[OK] Database URL found")
        else:
            print("[WARN]  Database URL not found. Set DATABASE_URL environment variable.")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        print(f"   Missing dependency. Install with: pip install {str(e).split()[-1]}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n[DEPS] Checking dependencies...")
    
    required_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'sqlalchemy',
        'requests',
        'python-dotenv',
        'fastapi',
        'slowapi'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'python-dotenv':
                __import__('dotenv')
            else:
                __import__(package.replace('-', '_'))
            print(f"[OK] {package}")
        except ImportError:
            print(f"[FAIL] {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[WARN]  Missing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    print("\n[OK] All dependencies available!")
    return True

if __name__ == "__main__":
    print("Enhanced Collaborative Filtering System - Import Test")
    print("=" * 60)
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if deps_ok:
        # Test imports
        imports_ok = test_imports()
        
        if imports_ok:
            print("\n[SUCCESS] System ready!")
            print("\nNext steps:")
            print("1. Set up environment variables (API_KEY, DATABASE_URL)")
            print("2. Run database migrations")
            print("3. Use the ingestion script to load your data:")
            print("   python -m scripts.library_ingestion_script --lastfm-username YOUR_USERNAME")
            sys.exit(0)
        else:
            print("\n[FAIL] Import test failed")
            sys.exit(1)
    else:
        print("\n[FAIL] Dependency check failed")
        sys.exit(1)