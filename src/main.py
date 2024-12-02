"""
Main application module that handles the initialization and running of the 3D viewer application.
This module coordinates between the Flask API server and the Pyglet graphics window.
"""

import logging
import threading
import esper
import pyglet
from pyglet.gl import *
from systems import RenderSystem
import api
from model_management import model_loader
from logging_config import setup_logging
from config import configure_assimp
from window_manager import window_manager
from world_manager import world_manager

# Configure assimp and logging at startup
configure_assimp()
setup_logging()
logger = logging.getLogger(__name__)

# Initialize ECS World
world = world_manager.world

class GameWindow(pyglet.window.Window):
    """
    Main application window that handles 3D rendering using Pyglet.
    
    This class manages the OpenGL context and coordinates with the render system
    to display 3D models. It inherits from pyglet.window.Window to provide
    basic window functionality.
    """

    def __init__(self):
        """Initialize the window with default size and setup OpenGL context."""
        super().__init__(800, 600, "ECS Render System", resizable=True)
        
        # Enable depth testing and set clear color to white
        glEnable(GL_DEPTH_TEST)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        
        # Additional GL settings for better rendering
        glEnable(GL_CULL_FACE)
        glShadeModel(GL_SMOOTH)
        
        # Initial projection setup
        self.update_projection(800, 600)
        
        self.render_system = RenderSystem(self)
        window_manager.window = self
        with world_manager.with_world():
            world.add_processor(self.render_system)

    def update_projection(self, width, height):
        """Update the projection matrix with the given dimensions."""
        if width == 0 or height == 0:
            return
        
        # Set viewport to full window dimensions
        glViewport(0, 0, width, height)
        
        # Setup projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Use gluPerspective for a more natural view
        aspect = width / float(height)
        gluPerspective(45.0, aspect, 0.1, 100.0)
        
        # Reset modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def on_draw(self):
        """Handle the window's draw event by clearing and rendering the scene."""
        self.clear()
        self.render_system.process()

    def on_resize(self, width, height):
        """Handle window resize events."""
        super().on_resize(width, height)  # Add this back to handle window system events
        if width == 0 or height == 0:
            return
        
        self.update_projection(width, height)

    def trigger_redraw(self):
        """Schedule a redraw of the window on the next frame."""
        with window_manager.with_window():
            pyglet.clock.schedule_once(lambda delta_time: None, 0)
            logger.debug("The window has been scheduled to redraw")

def run_flask_app():
    """
    Start the Flask API server in a separate thread.
    
    This function configures and runs the Flask application that handles
    HTTP requests for model manipulation.
    """
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.handlers = []
    werkzeug_logger.propagate = True

    logger.info("Starting Flask server...")
    try:
        api.app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
        logger.info("Flask server started successfully")
    except Exception as exception:
        logger.error(f"Failed to start Flask server: {str(exception)}", exc_info=True)

def run_pyglet():
    """
    Initialize and run the Pyglet event loop.
    
    This function sets up both the platform event loop and the main event loop
    required for proper Pyglet operation.
    """
    # Initialize platform event loop first
    pyglet.app.platform_event_loop = pyglet.app.PlatformEventLoop()
    pyglet.app.platform_event_loop.start()
    
    # Initialize event loop
    pyglet.app.event_loop = pyglet.app.EventLoop()
    
    # Start the event loop
    pyglet.app.event_loop.run()

if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    # Start Pyglet event loop in the main thread
    run_pyglet()

    # Keep the main thread alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")