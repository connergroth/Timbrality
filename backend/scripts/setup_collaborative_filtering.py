#!/usr/bin/env python3
"""
Script to set up collaborative filtering by adding multiple Last.fm users
and fetching their listening data.

Usage:
    python scripts/setup_collaborative_filtering.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.collaborative_filtering_service import CollaborativeFilteringService


async def main():
    """Main function to set up collaborative filtering"""
    print("🎵 Setting up Collaborative Filtering for Timbre")
    print("=" * 50)
    
    # Initialize the service
    service = CollaborativeFilteringService()
    
    # List of Last.fm usernames to add (you can modify this list)
    usernames = [
        "your_username",  # Replace with your actual username
        "sample_user_1",  # Add more usernames here
        "sample_user_2",
        # Add more users as needed...
    ]
    
    print(f"📋 Found {len(usernames)} usernames to process")
    print()
    
    # Step 1: Add users to the database
    print("🔧 Step 1: Adding users to database...")
    added_users = []
    
    for username in usernames:
        if username == "your_username":
            print(f"⚠️  Skipping placeholder username: {username}")
            continue
            
        print(f"  Adding user: {username}")
        result = await service.add_lastfm_user(username)
        
        if result["success"]:
            added_users.append(username)
            print(f"    ✅ Success: {result['message']}")
        else:
            print(f"    ❌ Failed: {result['message']}")
    
    print(f"✅ Successfully added {len(added_users)} users")
    print()
    
    if not added_users:
        print("❌ No users were added. Please check the usernames and try again.")
        return
    
    # Step 2: Fetch data for all users
    print("📥 Step 2: Fetching user data from Last.fm...")
    print("  This may take a while depending on the number of users...")
    
    try:
        results = await service.fetch_multiple_users_data(added_users)
        
        print(f"📊 Fetch Results Summary:")
        print(f"  Total users: {results['total_users']}")
        print(f"  Successful: {results['successful_fetches']}")
        print(f"  Failed: {results['failed_fetches']}")
        
        if results['errors']:
            print(f"  Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:  # Show first 3 errors
                print(f"    - {error}")
            if len(results['errors']) > 3:
                print(f"    ... and {len(results['errors']) - 3} more errors")
        
        print()
        
        # Step 3: Show active users
        print("👥 Step 3: Current active users:")
        active_users = await service.get_active_users()
        
        for user in active_users:
            print(f"  • {user['username']} ({user['display_name']})")
            if user['last_updated']:
                print(f"    Last updated: {user['last_updated']}")
        
        print()
        print("🎉 Collaborative filtering setup complete!")
        print()
        print("Next steps:")
        print("1. Check the database tables for collected data")
        print("2. Run similarity calculations between users")
        print("3. Generate collaborative filtering recommendations")
        print("4. Integrate with your recommendation engine")
        
    except Exception as e:
        print(f"❌ Error during data fetching: {str(e)}")
        print("Please check your Last.fm API configuration and try again.")


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("migrations/collaborative_filtering_setup.sql"):
        print("❌ Error: Please run this script from the backend directory")
        print("   cd backend && python scripts/setup_collaborative_filtering.py")
        sys.exit(1)
    
    # Run the async main function
    asyncio.run(main())




