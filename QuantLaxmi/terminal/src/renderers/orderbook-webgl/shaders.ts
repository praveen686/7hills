// ---------------------------------------------------------------------------
// WebGL2 GLSL shaders for the orderbook depth renderer
// ---------------------------------------------------------------------------

/** Vertex shader: transforms position and passes color to fragment shader. */
export const VERTEX_SHADER = `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in vec4 a_color;

uniform mat4 u_projection;

out vec4 v_color;

void main() {
  gl_Position = u_projection * vec4(a_position, 0.0, 1.0);
  v_color = a_color;
}
`;

/** Fragment shader: simple color passthrough. */
export const FRAGMENT_SHADER = `#version 300 es
precision mediump float;

in vec4 v_color;
out vec4 fragColor;

void main() {
  fragColor = v_color;
}
`;

/** Heatmap fragment shader: maps a scalar intensity to green-yellow-red gradient. */
export const HEATMAP_FRAGMENT = `#version 300 es
precision mediump float;

in vec4 v_color;
out vec4 fragColor;

void main() {
  float intensity = v_color.r; // intensity packed in red channel (0..1)

  // Green -> Yellow -> Red gradient
  vec3 color;
  if (intensity < 0.5) {
    float t = intensity * 2.0;
    color = mix(vec3(0.0, 0.83, 0.67), vec3(1.0, 0.72, 0.30), t); // profit -> warning
  } else {
    float t = (intensity - 0.5) * 2.0;
    color = mix(vec3(1.0, 0.72, 0.30), vec3(1.0, 0.30, 0.42), t); // warning -> loss
  }

  float alpha = 0.15 + intensity * 0.7;
  fragColor = vec4(color, alpha);
}
`;
