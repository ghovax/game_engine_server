import logging
from flask import Flask, request, jsonify
import pyglet
import sys
import threading
import esper
from enum import Enum
import numpy as np
from pyglet.gl import *
from pyglet.gl.gl import *
import pyassimp
import pyassimp.postprocess as postprocess
import os
import ctypes.util
import sqlite3
import urllib.request
import urllib.parse
import hashlib
from pathlib import Path
import datetime
from typing import Any, Dict

# Try multiple library paths
LIBRARY_PATHS = [
    "/opt/homebrew/Cellar/assimp/5.4.3/lib/libassimp.5.4.3.dylib",
    "/opt/homebrew/lib/libassimp.dylib",
    "/usr/local/lib/libassimp.dylib",
    "/usr/lib/libassimp.dylib",
]

# Set multiple environment variables to help find the library
for lib_path in LIBRARY_PATHS:
    if os.path.exists(lib_path):
        os.environ["ASSIMP_LIBRARY_PATH"] = lib_path
        os.environ["DYLD_LIBRARY_PATH"] = os.path.dirname(lib_path)
        break

# Also try to use ctypes to find the library
assimp_path = ctypes.util.find_library("assimp")
if assimp_path:
    os.environ["ASSIMP_PATH"] = assimp_path


class ColorCodes:
    GREY = "\x1b[38;21m"
    BLUE = "\x1b[38;5;39m"
    YELLOW = "\x1b[38;5;226m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"


class ColoredFormatter(logging.Formatter):
    def __init__(self, format_string):
        super().__init__(format_string)
        self.FORMATS = {
            logging.DEBUG: ColorCodes.GREY + format_string + ColorCodes.RESET,
            logging.INFO: ColorCodes.BLUE + format_string + ColorCodes.RESET,
            logging.WARNING: ColorCodes.YELLOW + format_string + ColorCodes.RESET,
            logging.ERROR: ColorCodes.RED + format_string + ColorCodes.RESET,
            logging.CRITICAL: ColorCodes.BOLD_RED + format_string + ColorCodes.RESET,
        }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


format_string = "%(levelname)s:%(name)s:%(message)s"
colored_formatter = ColoredFormatter(format_string)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(colored_formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])

# Get logger for this module
logger = logging.getLogger(__name__)

# Configure Flask's logger
flask_logger = logging.getLogger("werkzeug")
flask_logger.setLevel(logging.INFO)

# Configure PyAssimp logger
pyassimp_logger = logging.getLogger("pyassimp")
pyassimp_logger.setLevel(logging.INFO)

# Flask app setup
app = Flask(__name__)

# Add near the top of your file, after the imports
pyassimp_logger = logging.getLogger("pyassimp")
pyassimp_logger.setLevel(
    logging.INFO
)  # Can be altered for more granular logging if needed


# Define Components
class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Rotation:
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Color:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b


class Scale:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class ModelPath:
    def __init__(self, path):
        self.path = path


class ModelLoader:
    def __init__(self):
        self.loaded_models = {}
        self.supported_formats = {".obj", ".gltf", ".glb"}
        self.logger = logging.getLogger(__name__)

    def load_model(self, model_path: str) -> Dict[str, Any]:
        try:
            if model_path in self.loaded_models:
                self.logger.debug(f"Using cached version of model: {model_path}")
                return self.loaded_models[model_path]

            # Check if file exists
            if not os.path.exists(model_path):
                raise ValueError(f"Model file not found: {model_path}")

            extension = os.path.splitext(model_path)[1].lower()
            if extension not in self.supported_formats:
                raise ValueError(f"Unsupported format: {extension}")

            # For GLTF files, check if they're complete
            if extension == ".gltf":
                try:
                    with open(model_path, "r") as file:
                        import json

                        gltf_data = json.load(file)

                        # Check for external resources
                        if "buffers" in gltf_data:
                            for buffer in gltf_data["buffers"]:
                                if "uri" in buffer:
                                    buffer_path = os.path.join(
                                        os.path.dirname(model_path),
                                        buffer["uri"],
                                    )
                                    if not os.path.exists(buffer_path):
                                        raise ValueError(
                                            f"Missing binary file '{buffer['uri']}' required by this GLTF model, "
                                            "please ensure all .bin files are in the same directory as the .gltf file"
                                        )
                except json.JSONDecodeError:
                    raise ValueError("The GLTF file is not valid JSON")
                except Exception as exception:
                    if "Missing binary file" in str(exception):
                        raise
                    raise ValueError(f"Error processing GLTF file - {str(exception)}")

            self.logger.info(f"Loading model from {model_path}")
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
                    raise ValueError(f"No meshes found in model: {model_path}")

                self.logger.info(
                    f"Successfully loaded model with {len(scene.meshes)} meshes"
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
                                and mesh.texturecoords[0].size > 0
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
                f"Failed to load model '{os.path.basename(model_path)}' and "
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
            raise ValueError(f"Failed to load model due to unexpected error: {str(exception)}")

    def release_model(self, model_path):
        """
        Releases a model from memory.
        """
        if model_path in self.loaded_models:
            del self.loaded_models[model_path]
            self.logger.debug(f"Released model from memory: {model_path}")

    def __del__(self):
        """
        Cleanup method to release all loaded models when the loader is destroyed.
        """
        self.loaded_models.clear()


# Create global model loader instance
model_loader = ModelLoader()


# Define Systems
class RenderSystem(esper.Processor):
    def __init__(self, window):
        self.window = window

    def render_external_model(self, position, color, model_path, scale):
        scene = model_loader.load_model(model_path.path)
        if not scene:
            return

        glPushMatrix()
        glTranslatef(position.x, position.y, position.z)
        glScalef(scale.x, scale.y, scale.z)
        glColor3f(color.r, color.g, color.b)

        for mesh in scene["meshes"]:
            # Convert mesh data to numpy arrays
            vertices = np.array(mesh["vertices"], dtype=np.float32)
            normals = np.array(mesh["normals"], dtype=np.float32)

            # Enable arrays for vertices and normals
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)

            # Set up the vertex and normal arrays
            glVertexPointer(3, GL_FLOAT, 0, vertices.ctypes.data)
            glNormalPointer(GL_FLOAT, 0, normals.ctypes.data)

            # Draw the mesh
            glDrawArrays(GL_TRIANGLES, 0, len(vertices))

            # Disable arrays
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)

        glPopMatrix()

    def process(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)

        components = list(self.world.get_components(Position, Color))

        for entity, (position, color) in components:
            logger.debug(f"Rendering entity with ID {entity}")
            model_path = self.world.component_for_entity(entity, ModelPath)
            scale = self.world.component_for_entity(entity, Scale)
            if model_path and scale:
                self.render_external_model(position, color, model_path, scale)
            else:
                logger.error(f"Entity {entity} missing required components")

        glFlush()


# Initialize ECS World
world = esper.World()

# Modify window initialization and setup
window = None

# Global variable for tracking entities
next_entity_id = 1


# Utility function for successful responses
def success_response(data=None):
    return jsonify(data or {}), 200


# Utility function for error responses
def error_response(data=None, status_code=400):
    return jsonify(data or {}), status_code


class ModelDatabase:
    def __init__(self):
        self.db_path = "models.db"
        self.models_directory = Path("downloaded_models")
        self.models_directory.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    url TEXT PRIMARY KEY,
                    local_path TEXT NOT NULL,
                    last_modified TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    def get_local_path(self, url):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT local_path FROM models WHERE url = ?", (url,))
            result = cursor.fetchone()
            return result[0] if result else None

    def save_model_info(self, url, local_path):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO models (url, local_path) VALUES (?, ?)",
                (url, local_path),
            )


class ModelDownloader:
    def __init__(self):
        self.database = ModelDatabase()
        self.logger = logging.getLogger(__name__)

    def get_model_path(self, model_path: str) -> str:
        """Get or download a model path"""
        try:
            if not self._is_url(model_path):
                self.logger.debug(f"Using local path: {model_path}")
                return model_path

            self.logger.debug("Checking cache for model")
            local_path = self.database.get_local_path(model_path)
            if local_path and os.path.exists(local_path):
                self.logger.debug(f"Using cached version from: {local_path}")
                return local_path

            # Extract the file extension from the URL
            parsed_url = urllib.parse.urlparse(model_path)
            self.logger.debug(f"Parsed URL: {parsed_url}")

            original_extension = os.path.splitext(parsed_url.path)[1].lower()
            self.logger.debug(f"Detected file extension: {original_extension}")

            if not original_extension:
                self.logger.warning("No extension found in URL, defaulting to .obj")
                original_extension = ".obj"

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
            self.logger.debug(f"URL check for {path}: {is_url}")
            return is_url
        except Exception as exception:
            self.logger.error(
                f"Error checking if path is URL: {str(exception)}", exc_info=True
            )
            return False


def validate_entity_parameters(parameters):
    """Validates entity parameters and returns tuple (is_valid, error_message)"""
    required_fields = {
        "position": (list, 3),
        "objectColor": (list, 3),
        "rotation": (list, 3),
        "scale": (list, 3),
    }

    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in parameters]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Check field types and lengths
    for field, (expected_type, expected_length) in required_fields.items():
        value = parameters[field]

        # Check type
        if not isinstance(value, expected_type):
            return False, f"Field '{field}' must be of type {expected_type.__name__}"

        # Check length for lists
        if expected_length and len(value) != expected_length:
            return False, f"Field '{field}' must have exactly {expected_length} values"

        # Check numeric values for position, color, rotation, and scale
        if field in ["position", "objectColor", "rotation", "scale"]:
            if not all(isinstance(x, (int, float)) for x in value):
                return False, f"All values in '{field}' must be numbers"

            # Check color values are between 0 and 1
            if field == "objectColor" and not all(0 <= x <= 1 for x in value):
                return False, f"Color values must be between 0 and 1"

            # Check scale values are positive
            if field == "scale" and not all(x > 0 for x in value):
                return False, f"Scale values must be positive numbers"

    if "modelPath" not in parameters:
        return False, "modelPath is required for external models"
    if not isinstance(parameters["modelPath"], str):
        return False, "modelPath must be a string"
    if not parameters["modelPath"]:
        return False, "modelPath cannot be empty"

    return True, None


# Create global instances
model_downloader = ModelDownloader()


@app.route("/add_entity", methods=["POST"])
def add_entity():
    global next_entity_id

    try:
        logger.debug("Received new entity creation request")

        parameters = request.json
        if parameters is None:
            logger.error("Request contained invalid JSON data")
            return error_response({"error": "Invalid JSON"}, 400)

        # Validate parameters
        is_valid, error_message = validate_entity_parameters(parameters)
        if not is_valid:
            logger.error(f"Parameter validation failed: {error_message}")
            return error_response({"error": error_message}, 400)

        # Get local model path and try to load it
        try:
            local_model_path = model_downloader.get_model_path(parameters["modelPath"])
            model_loader.load_model(local_model_path)
        except ValueError as e:
            return error_response({"error": str(e)}, 400)
        except Exception as e:
            logger.error(f"Unexpected error processing model: {str(e)}")
            return error_response({"error": "Failed to process model"}, 500)

        # Create entity only after successful model loading
        entity = world.create_entity()

        # Add components that need unpacking
        for component_type, component_name in [
            (Position, "position"),
            (Color, "objectColor"),
            (Rotation, "rotation"),
            (Scale, "scale"),
        ]:
            world.add_component(entity, component_type(*parameters[component_name]))

        # Add the model path component
        world.add_component(entity, ModelPath(local_model_path))

        logger.info(
            f"Successfully created entity {entity} with model {local_model_path}"
        )
        next_entity_id += 1

        window.trigger_redraw()
        return success_response({"entityId": entity})

    except Exception as exception:
        logger.error(
            f"Failed to process entity creation request: {str(exception)}",
            exc_info=True,
        )
        return error_response({"error": "Internal server error"}, 500)


@app.route("/update_entity/<int:entity_id>", methods=["POST"])
def update_entity(entity_id):
    try:
        parameters = request.json
        logger.debug(
            f"Received update request for entity {entity_id} with parameters: {parameters}"
        )

        entity = world.get_entity(entity_id)
        if entity:
            # Update position
            if "position" in parameters:
                position = world.get_component(entity, Position)
                old_pos = [position.x, position.y, position.z]
                position.x, position.y, position.z = parameters["position"]
                logger.debug(
                    f"Updated position for entity {entity_id} from {old_pos} to {parameters['position']}"
                )

            # Update color
            if "objectColor" in parameters:
                color = world.get_component(entity, Color)
                old_color = [color.r, color.g, color.b]
                color.r, color.g, color.b = parameters["objectColor"]
                logger.debug(
                    f"Updated color for entity {entity_id} from {old_color} to {parameters['objectColor']}"
                )

            # Update rotation
            if "rotation" in parameters:
                rotation = world.get_component(entity, Rotation)
                old_rot = [rotation.pitch, rotation.yaw, rotation.roll]
                rotation.pitch, rotation.yaw, rotation.roll = parameters["rotation"]
                logger.debug(
                    f"Updated rotation for entity {entity_id} from {old_rot} to {parameters['rotation']}"
                )

            # Update scale
            if "scale" in parameters:
                scale = world.get_component(entity, Scale)
                old_scale = [scale.x, scale.y, scale.z]
                scale.x, scale.y, scale.z = parameters["scale"]
                logger.debug(
                    f"Updated scale for entity {entity_id} from {old_scale} to {parameters['scale']}"
                )

            logger.info(f"Successfully updated entity {entity_id}")
            window.trigger_redraw()
            return success_response({"entityId": entity_id})
        else:
            logger.error(f"Entity {entity_id} not found")
            return error_response({"error": "Entity not found"}, 404)

    except Exception as exception:
        logger.error(
            f"Error updating entity {entity_id}: {str(exception)}", exc_info=True
        )
        return error_response({"error": str(exception)}, 400)


@app.route("/remove_entity/<int:entity_id>", methods=["POST"])
def remove_entity(entity_id):
    try:
        logger.debug(f"Received request to remove entity {entity_id}")
        entity = world.get_entity(entity_id)
        if entity:
            # Log components being removed
            components = world.components_for_entity(entity)
            logger.debug(f"Removing entity {entity_id} with components: {components}")

            world.delete_entity(entity)
            logger.info(f"Successfully removed entity {entity_id}")

            window.trigger_redraw()
            return success_response(
                {"message": f"Entity {entity_id} removed successfully"}
            )
        else:
            logger.error(f"Entity {entity_id} not found")
            return error_response({"error": "Entity not found"}, 404)

    except Exception as exception:
        logger.error(
            f"Error removing entity {entity_id}: {str(exception)}", exc_info=True
        )
        return error_response({"error": str(exception)}, 400)


# Pyglet window setup
class GameWindow(pyglet.window.Window):
    def __init__(self):
        super().__init__(800, 600, "ECS Render System")
        glEnable(GL_DEPTH_TEST)  # Enable depth testing for 3D
        self.render_system = RenderSystem(self)
        # Add a reference to the window in the global scope
        global window
        window = self

    def on_draw(self):
        self.clear()
        self.render_system.process()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height, 1.0, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def trigger_redraw(self):
        # Schedule a redraw on the main thread since Pyglet is not thread-safe
        pyglet.clock.schedule_once(lambda dt: None, 0)


# Start Flask app in a separate thread
def run_flask_app():
    # Disable Flask's default logging handler
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.handlers = []
    werkzeug_logger.propagate = True

    logger.info("Starting Flask server...")
    try:
        # Listen on all interfaces (0.0.0.0) instead of just localhost
        app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
        logger.info("Flask server started successfully")
    except Exception as exception:
        logger.error(f"Failed to start Flask server: {str(exception)}", exc_info=True)


@app.route("/health", methods=["GET"])
def health_check():
    logger.debug("Health check endpoint called")
    return jsonify({"status": "ok"}), 200


# Main function
if __name__ == "__main__":
    # Initialize Pyglet and game window
    window = GameWindow()
    world.add_processor(window.render_system)

    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    # Run Pyglet's main loop
    pyglet.app.run()
