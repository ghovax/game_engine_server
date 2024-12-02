"""
API module that provides HTTP endpoints for interacting with the 3D viewer.

This module implements a REST API using Flask to allow external applications
to control the 3D viewer, including model loading, manipulation, and window management.
"""

import logging
from flask import Flask, request, jsonify
from components import Position, Color, Rotation, Scale, ModelPath
from model_management import model_loader, model_downloader
from window_manager import window_manager
import pyglet
import os

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable for tracking entities
next_entity_id = 1


def success_response(data=None):
    """
    Create a standardized success response.

    Args:
        data (dict, optional): Data to include in the response

    Returns:
        tuple: (JSON response, HTTP status code)
    """
    return jsonify(data or {}), 200


def error_response(data=None, status_code=400):
    """
    Create a standardized error response.

    Args:
        data (dict, optional): Error data to include in the response
        status_code (int, optional): HTTP status code to return

    Returns:
        tuple: (JSON response, HTTP status code)
    """
    return jsonify(data or {}), status_code


def validate_entity_parameters(parameters):
    required_fields = {
        "position": (list, 3),
        "objectColor": (list, 3),
        "rotation": (list, 3),
        "scale": (list, 3),
    }

    missing_fields = [field for field in required_fields if field not in parameters]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    for field, (expected_type, expected_length) in required_fields.items():
        value = parameters[field]

        if not isinstance(value, expected_type):
            return False, f"Field '{field}' must be of type {expected_type.__name__}"

        if expected_length and len(value) != expected_length:
            return False, f"Field '{field}' must have exactly {expected_length} values"

        if field in ["position", "objectColor", "rotation", "scale"]:
            if not all(isinstance(x, (int, float)) for x in value):
                return False, f"All values in '{field}' must be numbers"

            if field == "objectColor" and not all(0 <= x <= 1 for x in value):
                return False, f"Color values must be between 0 and 1"

            if field == "scale" and not all(x > 0 for x in value):
                return False, f"Scale values must be positive numbers"

    if "modelPath" not in parameters:
        return False, "modelPath is required for external models"
    if not isinstance(parameters["modelPath"], str):
        return False, "modelPath must be a string"
    if not parameters["modelPath"]:
        return False, "modelPath cannot be empty"

    return True, None


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
            model_id = model_downloader.database.get_model_id(local_model_path)
            model_id_str = f"#{model_id}" if model_id else os.path.basename(local_model_path)
            
            model_loader.load_model(local_model_path)
        except ValueError as exception:
            return error_response({"error": str(exception)}, 400)
        except Exception as exception:
            logger.error(f"Unexpected error processing model: {str(exception)}")
            return error_response({"error": "Failed to process model"}, 500)

        from main import world

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

        logger.info(f"Successfully created entity #{entity} with model {model_id_str}")
        next_entity_id += 1

        if window_manager.window:
            window_manager.window.trigger_redraw()
            logger.debug("Successfully triggered window redraw")
        else:
            logger.warning(
                f"Window is not available for redraw when adding entity #{entity}, probably not initialized yet, "
                "call the `init_window` endpoint first to initialize the window"
            )
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
        from main import world

        entity = world.get_entity(entity_id)
        if entity:
            if "position" in parameters:
                position = world.get_component(entity, Position)
                position.x, position.y, position.z = parameters["position"]

            if "objectColor" in parameters:
                color = world.get_component(entity, Color)
                color.r, color.g, color.b = parameters["objectColor"]

            if "rotation" in parameters:
                rotation = world.get_component(entity, Rotation)
                rotation.pitch, rotation.yaw, rotation.roll = parameters["rotation"]

            if "scale" in parameters:
                scale = world.get_component(entity, Scale)
                scale.x, scale.y, scale.z = parameters["scale"]

            logger.info(f"Successfully updated entity {entity_id}")
            from main import window

            if window:
                window.trigger_redraw()
            else:
                logger.error(
                    "Window is not available for redraw when updating entity "
                    f"{entity_id}"
                )
                raise

            return success_response({"entityId": entity_id})
        else:
            return error_response({"error": "Entity not found"}, 404)

    except Exception as exception:
        logger.error(
            f"Error updating entity {entity_id}: {str(exception)}", exc_info=True
        )
        return error_response({"error": str(exception)}, 400)


@app.route("/remove_entity/<int:entity_id>", methods=["POST"])
def remove_entity(entity_id):
    try:
        from main import world

        entity = world.get_entity(entity_id)
        if entity:
            # Log components being removed
            components = world.components_for_entity(entity)
            logger.debug(f"Removing entity {entity_id} with components: {components}")

            world.delete_entity(entity)
            logger.info(f"Successfully removed entity {entity_id}")

            from main import window

            if window:
                window.trigger_redraw()
            else:
                logger.error(
                    "Window is not available for redraw when removing entity "
                    f"{entity_id}"
                )
            return success_response(
                {"message": f"Entity {entity_id} removed successfully"}
            )
        else:
            return error_response({"error": "Entity not found"}, 404)

    except Exception as exception:
        logger.error(
            f"Error removing entity {entity_id}: {str(exception)}", exc_info=True
        )
        return error_response({"error": str(exception)}, 400)


@app.route("/health", methods=["GET"])
def health_check():
    logger.debug("Health check endpoint called")
    return jsonify({"status": "ok"}), 200


@app.route("/init_window", methods=["POST"])
def init_window():
    """
    Initialize the 3D viewer window.

    This endpoint creates a new window if one doesn't already exist.
    The window creation is scheduled on the Pyglet event loop to ensure
    thread safety.

    Returns:
        JSON response indicating success or failure
    """
    try:
        if window_manager.window is None:

            def create_window():
                from main import GameWindow

                window = GameWindow()
                logger.info("Graphics window initialized successfully")

            # Schedule window creation using pyglet's event loop
            pyglet.clock.schedule_once(lambda dt: create_window(), 0)
            return success_response({"message": "Window initialization scheduled"})
        else:
            return success_response({"message": "Window already initialized"})
    except Exception as exception:
        logger.error(f"Failed to initialize window: {str(exception)}", exc_info=True)
        return error_response({"error": str(exception)}, 500)
