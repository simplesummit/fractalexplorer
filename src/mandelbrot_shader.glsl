uniform vec2 center;
uniform float zoom;
uniform int max_iter;

void main() {
	vec2 z, z2, c;
    float fri;
    float ncl = 10;

	c.x = center.x + (2.0 * gl_TexCoord[0].x - 1.0) / zoom;
	c.y = center.y - (2.0 * gl_TexCoord[0].y - 1.0) / zoom;

	z = c;
    z2 = z * z;
	
    
    int ci = 0;

    while ((z2.x + z2.y < 16.0) && (ci < max_iter)) {
        z.y = 2.0 * z.x * z.y + c.y;
        z.x = z2.x - z2.y + c.x;
        z2 = z * z;
        ci++;
    }

    if (ci == max_iter) {
        fri = 0;
    } else {
        fri = mod(2 + ci - log(log(z2.x + z2.y)) / log(2.0f), ncl) / ncl;
    }

    gl_FragColor = vec4(fri, fri * fri, fri * fri * fri, 1.0);

}

