import { VERTEX_SHADER, HEATMAP_FRAGMENT } from "./shaders";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PriceLevel {
  price: number;
  size: number;
}

interface Snapshot {
  timestamp: number;
  bids: PriceLevel[];
  asks: PriceLevel[];
}

// ---------------------------------------------------------------------------
// Shader helpers (duplicated from OrderbookRenderer for independence)
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
// HeatmapRenderer — scrolling depth heatmap using historical snapshots
// ---------------------------------------------------------------------------

export class HeatmapRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private projLoc: WebGLUniformLocation | null;
  private history: Snapshot[] = [];
  private maxHistory = 200;
  private maxVertices = 50000;
  private vertexData: Float32Array;
  private width: number;
  private height: number;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;
    this.width = canvas.width;
    this.height = canvas.height;

    this.program = createProgram(gl, VERTEX_SHADER, HEATMAP_FRAGMENT);
    this.projLoc = gl.getUniformLocation(this.program, "u_projection");

    const vao = gl.createVertexArray();
    if (!vao) throw new Error("Failed to create VAO");
    this.vao = vao;

    const vbo = gl.createBuffer();
    if (!vbo) throw new Error("Failed to create VBO");
    this.vbo = vbo;

    this.vertexData = new Float32Array(this.maxVertices * 6);

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.vertexData.byteLength, gl.DYNAMIC_DRAW);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 24, 8);

    gl.bindVertexArray(null);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.031, 0.031, 0.051, 1.0);
  }

  /** Add a snapshot to the rolling history. */
  addSnapshot(ts: number, bids: PriceLevel[], asks: PriceLevel[]): void {
    this.history.push({ timestamp: ts, bids, asks });
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }
  }

  /** Render the scrolling heatmap from history. */
  render(): void {
    const { gl, width, height, history } = this;

    gl.viewport(0, 0, width, height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (history.length === 0) return;

    // Compute global price range and max size
    let globalMin = Infinity;
    let globalMax = -Infinity;
    let globalMaxSize = 0;

    for (const snap of history) {
      for (const lvl of [...snap.bids, ...snap.asks]) {
        if (lvl.price < globalMin) globalMin = lvl.price;
        if (lvl.price > globalMax) globalMax = lvl.price;
        if (lvl.size > globalMaxSize) globalMaxSize = lvl.size;
      }
    }

    const priceRange = globalMax - globalMin || 1;
    if (globalMaxSize === 0) globalMaxSize = 1;

    const colW = width / history.length;
    let offset = 0;

    const emitRect = (x1: number, y1: number, x2: number, y2: number, intensity: number) => {
      if (offset + 36 > this.vertexData.length) return;
      const verts = [
        x1, y1, intensity, 0, 0, 1,
        x2, y1, intensity, 0, 0, 1,
        x1, y2, intensity, 0, 0, 1,
        x1, y2, intensity, 0, 0, 1,
        x2, y1, intensity, 0, 0, 1,
        x2, y2, intensity, 0, 0, 1,
      ];
      this.vertexData.set(verts, offset);
      offset += 36;
    };

    // Render each snapshot as a vertical column of cells
    for (let col = 0; col < history.length; col++) {
      const snap = history[col];
      const x1 = col * colW;
      const x2 = x1 + colW;

      for (const lvl of [...snap.bids, ...snap.asks]) {
        const yNorm = (lvl.price - globalMin) / priceRange;
        const y = height - yNorm * height;
        const cellH = Math.max(height / 100, 2);
        const intensity = Math.min(lvl.size / globalMaxSize, 1.0);
        emitRect(x1, y - cellH / 2, x2, y + cellH / 2, intensity);
      }
    }

    // Upload and draw
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.projLoc, false, ortho(0, width, height, 0));

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
