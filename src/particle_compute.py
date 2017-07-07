"""
  particle_compute.py -- compute loop

  Copyright 2016-2017 ChemicalDevelopment

  This file is part of the fractalrender project.

  FractalRender source code, as well as any other resources in this project are
free software; you are free to redistribute it and/or modify them under
the terms of the GNU General Public License; either version 3 of the
license, or any later version.

  These programs are hopefully useful and reliable, but it is understood
that these are provided WITHOUT ANY WARRANTY, or MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GPLv3 or email at
<info@chemicaldevelopment.us> for more info on this.

  Here is a copy of the GPL v3, which this software is licensed under. You
can also find a copy at http://www.gnu.org/licenses/.
"""


from OpenGLContext import testingcontext
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGLContext.arrays import *
from OpenGL.GL import shaders
BaseContext = testingcontext.getInteractive()

class TestContext( BaseContext ):
    """Creates a simple vertex shader..."""
    def OnInit( self ):
        VERTEX_SHADER = shaders.compileShader("""#version 120
        void main() {
            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
        }""", GL_VERTEX_SHADER)
        FRAGMENT_SHADER = shaders.compileShader("""#version 120
        void main() {
            gl_FragColor = vec4( 0, 1, 0, 1 );
        }""", GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
        """self.vbo = vbo.VBO(
            array( [
                [  0, 1, 0 ],
                [ -1,-1, 0 ],
                [  1,-1, 0 ],
                [  2,-1, 0 ],
                [  4,-1, 0 ],
                [  4, 1, 0 ],
                [  2,-1, 0 ],
                [  4, 1, 0 ],
                [  2, 1, 0 ],
            ],'f')
        )"""

    def Render( self, mode):
        """Render the geometry for the scene."""
        shaders.glUseProgram(self.shader)
        #self.vbo.bind()
        glBegin(GL_TRIANGLE_FAN);
        #glEnableClientState(GL_VERTEX_ARRAY);
        #glVertexPointerf( self.vbo )
        glVertexPoint3f(0, 1, 0)
        glVertexPoint3f(-1, -1, 0)
        glVertexPoint3f(1, -1, 0)
        #glDrawArrays(GL_TRIANGLES, 0, 9)
        glEnd()
        #glDisableClientState(GL_VERTEX_ARRAY);

def main():
    TestContext.ContextMainLoop()
