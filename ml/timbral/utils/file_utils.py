"""
File utility functions for Timbral.

This module provides utility functions for common file operations
used throughout the music recommendation system.
"""

import os
import json
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FileUtils:
    """
    File utility functions for common file operations.
    
    This class provides methods for loading, saving, and managing
    files in various formats used by the recommendation system.
    """
    
    def __init__(self):
        """
        Initialize the file utilities.
        """
        pass
    
    def ensure_directory(self, directory_path: str) -> bool:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Success status
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            return False
    
    def load_json(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Loaded data or None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON from {filepath}: {e}")
            return None
    
    def save_json(
        self,
        data: Dict[str, Any],
        filepath: str,
        indent: int = 2
    ) -> bool:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            filepath: Path to save file
            indent: JSON indentation
            
        Returns:
            Success status
        """
        try:
            self.ensure_directory(os.path.dirname(filepath))
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON to {filepath}: {e}")
            return False
    
    def load_yaml(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load data from YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Loaded data or None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load YAML from {filepath}: {e}")
            return None
    
    def save_yaml(
        self,
        data: Dict[str, Any],
        filepath: str,
        default_flow_style: bool = False
    ) -> bool:
        """
        Save data to YAML file.
        
        Args:
            data: Data to save
            filepath: Path to save file
            default_flow_style: YAML flow style
            
        Returns:
            Success status
        """
        try:
            self.ensure_directory(os.path.dirname(filepath))
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=default_flow_style)
            return True
        except Exception as e:
            logger.error(f"Failed to save YAML to {filepath}: {e}")
            return False
    
    def load_pickle(self, filepath: str) -> Optional[Any]:
        """
        Load data from pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Loaded data or None
        """
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle from {filepath}: {e}")
            return None
    
    def save_pickle(
        self,
        data: Any,
        filepath: str,
        protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> bool:
        """
        Save data to pickle file.
        
        Args:
            data: Data to save
            filepath: Path to save file
            protocol: Pickle protocol version
            
        Returns:
            Success status
        """
        try:
            self.ensure_directory(os.path.dirname(filepath))
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
            return True
        except Exception as e:
            logger.error(f"Failed to save pickle to {filepath}: {e}")
            return False
    
    def get_file_size(self, filepath: str) -> Optional[int]:
        """
        Get file size in bytes.
        
        Args:
            filepath: Path to file
            
        Returns:
            File size in bytes or None
        """
        try:
            return os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Failed to get file size for {filepath}: {e}")
            return None
    
    def file_exists(self, filepath: str) -> bool:
        """
        Check if file exists.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file exists, False otherwise
        """
        return os.path.isfile(filepath)
    
    def directory_exists(self, directory_path: str) -> bool:
        """
        Check if directory exists.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            True if directory exists, False otherwise
        """
        return os.path.isdir(directory_path)
    
    def list_files(
        self,
        directory_path: str,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[str]:
        """
        List files in directory matching pattern.
        
        Args:
            directory_path: Path to directory
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        try:
            path = Path(directory_path)
            if recursive:
                files = path.rglob(pattern)
            else:
                files = path.glob(pattern)
            return [str(f) for f in files if f.is_file()]
        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {e}")
            return []
    
    def copy_file(
        self,
        source_path: str,
        destination_path: str
    ) -> bool:
        """
        Copy file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            Success status
        """
        try:
            import shutil
            self.ensure_directory(os.path.dirname(destination_path))
            shutil.copy2(source_path, destination_path)
            return True
        except Exception as e:
            logger.error(f"Failed to copy file from {source_path} to {destination_path}: {e}")
            return False
    
    def delete_file(self, filepath: str) -> bool:
        """
        Delete a file.
        
        Args:
            filepath: Path to file to delete
            
        Returns:
            Success status
        """
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {e}")
            return False 