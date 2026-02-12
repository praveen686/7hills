import { VERTEX_SHADER, FRAGMENT_SHADER } from "./shaders";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PriceLevel {
  price: number;
  size: number;
}

// ---------------------------------------------------------------------------
// Shader compilation helpers
// ---------------------------------------------------------------------------

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Failed to create shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${info}`);
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const vert = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
  const program = gl.createProgram();
  if (!program) throw new Error("Failed to create program");
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${info}`);
  }
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  return program;
}

// ---------------------------------------------------------------------------
// Orthographic projection matrix (column-major for WebGL)
// ---------------------------------------------------------------------------

function ortho(left: number, right: number, bottom: number, top: number): Float32Array {
  const lr = 1 / (right - left);
  const bt = 1 / (top - bottom);
  return new Float32Array([
    2 * lr, 0, 0, 0,
    0, 2 * bt, 0, 0,
    0, 0, -1, 0,
    -(right + left) * lr, -(top + bottom) * bt, 0, 1,
  ]);
}

// ---------------------------------------------------------------------------
// OrderbookRenderer class
// ---------------------------------------------------------------------------

export class OrderbookRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private projLoc: WebGLUniformLocation | null;
  private maxVertices = 2048;
  private vertexData: Float32Array;
  private width: number;
  private height: number;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;
    this.width = canvas.width;
    this.height = canvas.height;

    // Compile shaders and create program
    this.program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    this.projLoc = gl.getUniformLocation(this.program, "u_projection");

    // Create VAO and VBO
    const vao = gl.createVertexArray();
    if (!vao) throw new Error("Failed to create VAO");
    this.vao = vao;

    const vbo = gl.createBuffer();
    if (!vbo) throw new Error("Failed to create VBO");
    this.vbo = vbo;

    // 6 floats per vertex: x, y, r, g, b, a
    this.vertexData = new Float32Array(this.maxVertices * 6);

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertexData.byteLength, gl.DYNAMIC_DRAW);

    // a_position (location 0): 2 floats
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);

    // a_color (location 1): 4 floats
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 8);

    gl.bindVertexArray(null);

    // GL state
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.031, 0.031, 0.051, 1.0); // terminal-bg
  }

  /** Render bid/ask depth bars. */
  render(bids: PriceLevel[], asks: PriceLevel[]): void {
    const { gl, width, height } = this;

    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (bids.length === 0 && asks.length === 0) return;

    // Find price range and max cumulative
    const allPrices = [...bids, ...asks].map((l) => l.price);
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const priceRange = maxPrice - minPrice || 1;

    let maxCum = 0;
    let cum = 0;
    for (const b of bids) { cum += b.size; maxCum = Math.max(maxCum, cum); }
    cum = 0;
    for (const a of asks) { cum += a.size; maxCum = Math.max(maxCum, cum); }
    if (maxCum === 0) maxCum = 1;

    // Build vertex data: quads (2 triangles = 6 vertices each)
    let offset = 0;
    const halfW = width / 2;

    // Helper to emit a rect
    const emitRect = (x1: number, y1: number, x2: number, y2: number, r: number, g: number, b: number, a: number) => {
      if (offset + 36 > this.vertexData.length) return;
      const verts = [
        x1, y1, r, g, b, a,
        x2, y1, r, g, b, a,
        x1, y2, r, g, b, a,
        x1, y2, r, g, b, a,
        x2, y1, r, g, b, a,
        x2, y2, r, g, b, a,
      ];
      this.vertexData.set(verts, offset);
      offset += 36;
    };

    // Bids — green bars, right side
    cum = 0;
    for (const bid of bids) {
      cum += bid.size;
      const y = height - ((bid.price - minPrice) / priceRange) * height;
      const barW = (cum / maxCum) * halfW;
      emitRect(halfW, y - 2, halfW + barW, y + 2, 0, 0.83, 0.67, 0.5);
    }

    // Asks — red bars, left side
    cum = 0;
    for (const ask of asks) {
      cum += ask.size;
      const y = height - ((ask.price - minPrice) / priceRange) * height;
      const barW = (cum / maxCum) * halfW;
      emitRect(halfW - barW, y - 2, halfW, y + 2, 1, 0.30, 0.42, 0.5);
    }

    // Upload and draw
    gl.useProgram(this.program);

    const proj = ortho(0, width, height, 0);
    gl.uniformMatrix4fv(this.projLoc, false, proj);

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.vertexData.subarray(0, offset));
    gl.drawArrays(gl.TRIANGLES, 0, offset / 6);
    gl.bindVertexArray(null);
  }

  /** Handle canvas resize. */
  resize(w: number, h: number): void {
    this.width = w;
    this.height = h;
    this.gl.canvas.width = w;
    this.gl.canvas.height = h;
  }

  /** Clean up GPU resources. */
  destroy(): void {
    const { gl } = this;
    gl.deleteBuffer(this.vbo);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
  }
}
