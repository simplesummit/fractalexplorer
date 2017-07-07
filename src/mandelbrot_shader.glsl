uniform vec2 center;
uniform float zoom;
uniform float er2;
uniform int max_iter;

void main() {
	vec2 z, z2, c;

	c.x = center.x - (2.0 * gl_TexCoord[0].x - 1.0) / zoom;
	c.y = center.y + (2.0 * gl_TexCoord[0].y - 1.0) / zoom;

	z = c;
        z2 = z * z;

	int ci;
        while ((z2.x + z2.y < er2) && (ci < max_iter)) {
            z.y = 2.0 * z.x * z.y + c.x;
            z.x = z2.x - z2.y + c.y;
            z2 = z * z;
            ci++;
        }

	gl_FragColor = vec4(0, 0, 1.0, 1.0);
}

