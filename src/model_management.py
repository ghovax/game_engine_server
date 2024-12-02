import os
import logging
import sqlite3
import urllib.request
import urllib.parse
import hashlib
from pathlib import Path
import pyassimp
import pyassimp.postprocess as postprocess
from typing import Any, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.loaded_models = {}
        self.supported_formats = {".obj", ".gltf", ".glb"}
        self.logger = logging.getLogger(__name__)
        self.logged_cache_hits = set()

    def load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            # Get model ID from database
            model_id = model_downloader.database.get_model_id(model_path)
            model_id_str = f"#{model_id}" if model_id else os.path.basename(model_path)

            if model_path in self.loaded_models:
                if model_path not in self.logged_cache_hits:
                    self.logger.debug(f"Using cached version of model: {model_id_str}")
                    self.logged_cache_hits.add(model_path)
                return self.loaded_models[model_path]

            self.logger.info(f"Loading model {model_id_str}")
            processing_flags = (
                postprocess.aiProcess_Triangulate
                | postprocess.aiProcess_GenNormals
                | postprocess.aiProcess_GenUVCoords
                | postprocess.aiProcess_OptimizeMeshes
            )

            with pyassimp.load(model_path, processing=processing_flags) as scene:
                if not scene:
                    raise ValueError("Failed to load scene - scene is None")
                if not scene.meshes:
                    raise ValueError(f"No meshes found in model: {model_id_str}")

                self.logger.info(
                    f"Successfully loaded model {model_id_str} with {len(scene.meshes)} meshes"
                )

                cached_scene = {
                    "meshes": [
                        {
                            "vertices": mesh.vertices.copy(),
                            "normals": (
                                mesh.normals.copy()
                                if mesh.normals is not None
                                else None
                            ),
                            "faces": (
                                mesh.faces.copy() if hasattr(mesh, "faces") else None
                            ),
                            "texturecoords": (
                                mesh.texturecoords[0].copy()
                                if mesh.texturecoords is not None
                                and len(mesh.texturecoords) > 0
                                else None
                            ),
                        }
                        for mesh in scene.meshes
                    ]
                }
                self.loaded_models[model_path] = cached_scene
                return cached_scene

        except pyassimp.errors.AssimpError as exception:
            error_message = (
                f"Failed to load model '{model_id_str}' and "
                "this could be because the required files (textures, .bin files) are missing "
                "or the model uses features not supported by the importer"
            )
            self.logger.error(error_message)
            raise ValueError(error_message)
        except ValueError as exception:
            # Pass through our custom error messages without modification
            self.logger.error(str(exception))
            raise
        except Exception as exception:
            self.logger.error(f"Unexpected error loading model: {str(exception)}")
            raise ValueError(
                f"Failed to load model due to unexpected error: {str(exception)}"
            )

    def release_model(self, model_path):
        """
        Releases a model from memory.
        """
        try:
            model_filename = os.path.basename(model_path)
            if model_path in self.loaded_models:
                del self.loaded_models[model_path]
                self.logger.debug(f"Released model from memory: {model_filename}")
            else:
                self.logger.warning(f"Model not found in loaded models: {model_filename}")
        except Exception as exception:
            self.logger.error(f"Unexpected error releasing model: {str(exception)}")
            raise ValueError(
                f"Failed to release model due to unexpected error: {str(exception)}"
            )

    def __del__(self):
        """
        Cleanup method to release all loaded models when the loader is destroyed.
        """
        self.loaded_models.clear()


class ModelDatabase:
    def __init__(self):
        self.database_path = "../models.db"
        self.models_directory = Path("../downloaded_models")
        self.models_directory.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        with sqlite3.connect(self.database_path) as connection:
            # Add model_id column as autoincrementing primary key
            connection.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    local_path TEXT NOT NULL,
                    original_filename TEXT,
                    file_size INTEGER,
                    last_modified TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    download_count INTEGER DEFAULT 0
                )
            """)

    def get_model_info(self, url):
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute("""
                SELECT 
                    model_id,
                    local_path, 
                    original_filename,
                    file_size,
                    last_modified,
                    created_at,
                    last_accessed,
                    download_count
                FROM models 
                WHERE url = ?
            """, (url,))
            result = cursor.fetchone()
            if result:
                return {
                    'model_id': result[0],
                    'local_path': result[1],
                    'original_filename': result[2],
                    'file_size': result[3],
                    'last_modified': result[4],
                    'created_at': result[5],
                    'last_accessed': result[6],
                    'download_count': result[7]
                }
            return None

    def get_model_id(self, local_path):
        """Get model ID from local path"""
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute("""
                SELECT model_id FROM models WHERE local_path = ?
            """, (local_path,))
            result = cursor.fetchone()
            return result[0] if result else None

    def update_access(self, url):
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("""
                UPDATE models 
                SET last_accessed = CURRENT_TIMESTAMP,
                    download_count = download_count + 1
                WHERE url = ?
            """, (url,))

    def save_model_info(self, url, local_path, original_filename=None):
        file_size = os.path.getsize(local_path) if os.path.exists(local_path) else None
        last_modified = datetime.fromtimestamp(os.path.getmtime(local_path)).isoformat() if os.path.exists(local_path) else None
        
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute("""
                INSERT INTO models (
                    url, local_path, original_filename, file_size, last_modified
                ) VALUES (?, ?, ?, ?, ?)
            """, (url, local_path, original_filename, file_size, last_modified))
            return cursor.lastrowid


class ModelDownloader:
    def __init__(self):
        self.database = ModelDatabase()
        self.logger = logging.getLogger(__name__)

    def get_model_path(self, model_path: str) -> str:
        """Get or download a model path"""
        try:
            if not self._is_url(model_path):
                model_filename = os.path.basename(model_path)
                self.logger.debug(f"Using local path: {model_filename}")
                return model_path

            self.logger.debug("Checking cache for model")
            model_info = self.database.get_model_info(model_path)
            
            if model_info and os.path.exists(model_info['local_path']):
                model_filename = os.path.basename(model_info['local_path'])
                self.logger.info(f"Using cached model: {model_filename}")
                self.database.update_access(model_path)
                return model_info['local_path']

            # Extract the file extension from the URL
            parsed_url = urllib.parse.urlparse(model_path)
            self.logger.debug(f"Parsed URL: {parsed_url}")

            original_extension = os.path.splitext(parsed_url.path)[1].lower()
            self.logger.debug(f"Detected file extension: {original_extension}")

            if not original_extension:
                # TODO: Add support for other file types in the request
                raise ValueError("No extension found in URL")

            # Create filename preserving the original extension
            filename = hashlib.md5(model_path.encode()).hexdigest() + original_extension
            local_path = str(self.database.models_directory / filename)
            self.logger.debug(f"Created download path: {local_path}")

            self._download_file(model_path, local_path)
            self.database.save_model_info(model_path, local_path)
            self.logger.info(f"Successfully downloaded model to: {local_path}")
            return local_path

        except Exception as exception:
            self.logger.error(
                f"Failed to download model: {str(exception)}", exc_info=True
            )
            raise

    def _download_file(self, url, local_path):
        """Helper method to download a single file"""
        self.logger.debug(f"Starting download from {url} to {local_path}")
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }

            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP Error: {response.status}")

                content_length = response.headers.get("content-length", "unknown")
                self.logger.debug(f"Download started, content length: {content_length}")

                with open(local_path, "wb") as file:
                    file.write(response.read())
                self.logger.debug("Download completed successfully")

        except Exception as exception:
            self.logger.error(f"Download failed: {str(exception)}", exc_info=True)
            raise

    def _is_url(self, path):
        try:
            result = urllib.parse.urlparse(path)
            is_url = all([result.scheme, result.netloc])
            self.logger.debug(f"URL {'exists' if is_url else 'does not exist'} for {path}")
            return is_url
        except Exception as exception:
            self.logger.error(
                f"Error checking if path is URL: {str(exception)}", exc_info=True
            )
            return False


# Create global instances
model_loader = ModelLoader()
model_downloader = ModelDownloader()
