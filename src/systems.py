import logging
import numpy as np
from pyglet.gl import *
import esper
from components import Position, Color, ModelPath, Scale, Rotation
import os

logger = logging.getLogger(__name__)


class RenderSystem(esper.Processor):
    def __init__(self, window):
        self.window = window
        self.logged_models = set()  # Track which models we've logged info for
        self.scaling_factor = 2

    def setup_projection(self):
        """Setup projection matrix with correct aspect ratio."""
        # Ensure OpenGL uses the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Calculate aspect ratio and set perspective projection
        width, height = self.window.width, self.window.height
        aspect_ratio = width / height if height != 0 else 1  # Avoid division by zero
        gluPerspective(45.0, aspect_ratio, 0.01, 1000.0)

        # Return to modelview matrix for further transformations
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def setup_camera(self):
        """Setup camera position and lighting."""
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.0, 1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # Configure lighting
        light_position = (GLfloat * 4)(5.0, 5.0, 5.0, 1.0)
        ambient_light = (GLfloat * 4)(0.4, 0.4, 0.4, 1.0)
        diffuse_light = (GLfloat * 4)(0.8, 0.8, 0.8, 1.0)
        specular_light = (GLfloat * 4)(1.0, 1.0, 1.0, 1.0)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular_light)

        global_ambient = (GLfloat * 4)(0.2, 0.2, 0.2, 1.0)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, global_ambient)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)

        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_NORMALIZE)

    def render_external_model(self, position, color, rotation, model_path, scale):
        from model_management import model_loader, model_downloader

        scene = model_loader.load_model(model_path.path)
        if not scene:
            return

        # Get model ID for logging
        model_id = model_downloader.database.get_model_id(model_path.path)
        model_id_str = f"#{model_id}" if model_id else os.path.basename(model_path.path)

        # Only log model information once
        if model_path.path not in self.logged_models:
            for mesh in scene["meshes"]:
                if mesh["normals"] is None:
                    logger.warning(
                        f"Model {model_id_str} has no normals - lighting won't work properly"
                    )
                    break
                else:
                    logger.debug(
                        f"Model {model_id_str} has {len(mesh['normals'])} normal vectors"
                    )
            self.logged_models.add(model_path.path)

        glPushMatrix()

        # Apply transformations
        glTranslatef(position.x, position.y, position.z)
        glRotatef(rotation.yaw, 0, 1, 0)
        glRotatef(rotation.pitch, 1, 0, 0)
        glRotatef(rotation.roll, 0, 0, 1)
        glScalef(scale.x, scale.y, scale.z)

        # Set material properties
        glColor3f(color.r, color.g, color.b)  # Base color

        # Material properties
        ambient = (GLfloat * 4)(color.r * 0.4, color.g * 0.4, color.b * 0.4, 1.0)
        diffuse = (GLfloat * 4)(color.r, color.g, color.b, 1.0)
        specular = (GLfloat * 4)(0.5, 0.5, 0.5, 1.0)
        shininess = 64.0

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)

        # Draw the model
        for mesh in scene["meshes"]:
            vertices = np.array(mesh["vertices"], dtype=np.float32)
            normals = (
                np.array(mesh["normals"], dtype=np.float32)
                if mesh["normals"] is not None
                else None
            )
            faces = mesh["faces"]

            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, vertices.ctypes.data)

            if normals is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, normals.ctypes.data)

            if faces is not None:
                faces = np.array(faces, dtype=np.uint32)
                glDrawElements(
                    GL_TRIANGLES,
                    len(faces.flatten()),
                    GL_UNSIGNED_INT,
                    faces.ctypes.data,
                )
            else:
                glDrawArrays(GL_TRIANGLES, 0, len(vertices))

            # Cleanup states
            glDisableClientState(GL_VERTEX_ARRAY)
            if normals is not None:
                glDisableClientState(GL_NORMAL_ARRAY)

        glPopMatrix()

    def process(self):
        """Render all entities in the scene."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # TODO: Get the scaling factor for the display automatically, so we can make sure the model
        # is the same size on all displays and the viewport covers the entire window
        glViewport(
            0,
            0,
            self.window.width * self.scaling_factor,
            self.window.height * self.scaling_factor,
        )

        # Setup projection and camera
        self.setup_projection()
        self.setup_camera()

        # Render entities
        components = self.world.get_components(
            Position, Color, Rotation, ModelPath, Scale
        )
        for entity, (position, color, rotation, model_path, scale) in components:
            try:
                self.render_external_model(position, color, rotation, model_path, scale)
            except Exception as e:
                logger.error(f"Failed to render entity #{entity}: {str(e)}")

        # Ensure all rendering is completed
        glFlush()
