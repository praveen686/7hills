// ---------------------------------------------------------------------------
// DOM Ladder Renderer — Canvas 2D based price ladder
// ---------------------------------------------------------------------------

interface PriceLevel {
  price: number;
  size: number;
}

interface LadderClickEvent {
  price: number;
  side: "BUY" | "SELL";
}

type LadderClickHandler = (event: LadderClickEvent) => void;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GRID_COLOR = "#1e1e2e";
const TEXT_COLOR = "#c8c8d4";
const MUTED_COLOR = "#6b6b8a";
const BID_COLOR = "rgba(0, 212, 170, ";
const ASK_COLOR = "rgba(255, 77, 106, ";
const CENTER_BG = "#13131d";
const FONT = "11px 'JetBrains Mono', monospace";
const SMALL_FONT = "9px 'JetBrains Mono', monospace";
const ROW_HEIGHT = 22;
const PRICE_COL_W = 90;
const SIZE_COL_W = 70;

// ---------------------------------------------------------------------------
// LadderRenderer class
// ---------------------------------------------------------------------------

export class LadderRenderer {
  private ctx: CanvasRenderingContext2D;
  private canvas: HTMLCanvasElement;
  private width: number;
  private height: number;
  private clickHandler: LadderClickHandler | null = null;
  private lastBids: PriceLevel[] = [];
  private lastAsks: PriceLevel[] = [];
  private lastCenter = 0;
  private lastTick = 0;
  private boundClick: (e: MouseEvent) => void;

  constructor(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D not supported");
    this.ctx = ctx;
    this.canvas = canvas;
    this.width = canvas.width;
    this.height = canvas.height;

    // Bind click handler
    this.boundClick = this.handleCanvasClick.bind(this);
    canvas.addEventListener("click", this.boundClick);
  }

  /** Set click handler for price level clicks. */
  set onClick(handler: LadderClickHandler | null) {
    this.clickHandler = handler;
  }

  /** Render the price ladder centered on the given price. */
  render(
    centerPrice: number,
    bids: PriceLevel[],
    asks: PriceLevel[],
    tickSize: number,
  ): void {
    this.lastBids = bids;
    this.lastAsks = asks;
    this.lastCenter = centerPrice;
    this.lastTick = tickSize;

    const { ctx, width, height } = this;
    const dpr = window.devicePixelRatio || 1;

    // Set canvas actual size for HiDPI
    if (this.canvas.width !== width * dpr || this.canvas.height !== height * dpr) {
      this.canvas.width = width * dpr;
      this.canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, width, height);

    // Calculate visible rows
    const visibleRows = Math.floor(height / ROW_HEIGHT);
    const halfRows = Math.floor(visibleRows / 2);

    // Build price levels to display
    const bidMap = new Map(bids.map((l) => [l.price, l.size]));
    const askMap = new Map(asks.map((l) => [l.price, l.size]));

    // Max size for bar scaling
    const maxSize = Math.max(
      ...bids.map((l) => l.size),
      ...asks.map((l) => l.size),
      1,
    );

    // Column layout: [bid size] [price] [ask size]
    const bidColX = 0;
    const priceColX = SIZE_COL_W;
    const askColX = SIZE_COL_W + PRICE_COL_W;

    // Draw each row
    for (let row = 0; row < visibleRows; row++) {
      const priceOffset = halfRows - row;
      const price = centerPrice + priceOffset * tickSize;
      const y = row * ROW_HEIGHT;
      const yMid = y + ROW_HEIGHT / 2 + 4;

      const bidSize = bidMap.get(price) ?? 0;
      const askSize = askMap.get(price) ?? 0;
      const isCenter = priceOffset === 0;

      // Row background
      if (isCenter) {
        ctx.fillStyle = CENTER_BG;
        ctx.fillRect(0, y, width, ROW_HEIGHT);
      }

      // Grid line
      ctx.strokeStyle = GRID_COLOR;
      ctx.beginPath();
      ctx.moveTo(0, y + ROW_HEIGHT);
      ctx.lineTo(width, y + ROW_HEIGHT);
      ctx.stroke();

      // Bid size bar (right-aligned in bid column)
      if (bidSize > 0) {
        const barW = (bidSize / maxSize) * SIZE_COL_W * 0.85;
        ctx.fillStyle = BID_COLOR + "0.35)";
        ctx.fillRect(bidColX + SIZE_COL_W - barW, y + 2, barW, ROW_HEIGHT - 4);

        ctx.fillStyle = TEXT_COLOR;
        ctx.font = SMALL_FONT;
        ctx.textAlign = "right";
        ctx.fillText(this.formatSize(bidSize), bidColX + SIZE_COL_W - 4, yMid);
      }

      // Price label
      ctx.fillStyle = isCenter ? "#4f8eff" : (bidSize > 0 ? "#00d4aa" : askSize > 0 ? "#ff4d6a" : MUTED_COLOR);
      ctx.font = FONT;
      ctx.textAlign = "center";
      ctx.fillText(price.toFixed(2), priceColX + PRICE_COL_W / 2, yMid);

      // Ask size bar (left-aligned in ask column)
      if (askSize > 0) {
        const barW = (askSize / maxSize) * SIZE_COL_W * 0.85;
        ctx.fillStyle = ASK_COLOR + "0.35)";
        ctx.fillRect(askColX, y + 2, barW, ROW_HEIGHT - 4);

        ctx.fillStyle = TEXT_COLOR;
        ctx.font = SMALL_FONT;
        ctx.textAlign = "left";
        ctx.fillText(this.formatSize(askSize), askColX + 4, yMid);
      }
    }

    // Column dividers
    ctx.strokeStyle = GRID_COLOR;
    ctx.beginPath();
    ctx.moveTo(SIZE_COL_W, 0);
    ctx.lineTo(SIZE_COL_W, height);
    ctx.moveTo(SIZE_COL_W + PRICE_COL_W, 0);
    ctx.lineTo(SIZE_COL_W + PRICE_COL_W, height);
    ctx.stroke();

    // Column headers
    ctx.fillStyle = MUTED_COLOR;
    ctx.font = SMALL_FONT;
    ctx.textAlign = "center";
    ctx.fillText("BID", SIZE_COL_W / 2, 12);
    ctx.fillText("PRICE", SIZE_COL_W + PRICE_COL_W / 2, 12);
    ctx.fillText("ASK", askColX + SIZE_COL_W / 2, 12);
  }

  /** Handle canvas resize. */
  resize(w: number, h: number): void {
    this.width = w;
    this.height = h;
    // Re-render if we have data
    if (this.lastCenter > 0 && this.lastTick > 0) {
      this.render(this.lastCenter, this.lastBids, this.lastAsks, this.lastTick);
    }
  }

  /** Clean up event listeners. */
  destroy(): void {
    this.canvas.removeEventListener("click", this.boundClick);
  }

  // ---- Private helpers ----

  private handleCanvasClick(e: MouseEvent): void {
    if (!this.clickHandler || this.lastTick === 0) return;

    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const row = Math.floor(y / ROW_HEIGHT);
    const visibleRows = Math.floor(this.height / ROW_HEIGHT);
    const halfRows = Math.floor(visibleRows / 2);
    const priceOffset = halfRows - row;
    const price = this.lastCenter + priceOffset * this.lastTick;

    // Left half = bid side, right half = ask side
    const side: "BUY" | "SELL" = x < this.width / 2 ? "BUY" : "SELL";

    this.clickHandler({ price, side });
  }

  private formatSize(size: number): string {
    if (size >= 100000) return `${(size / 1000).toFixed(0)}K`;
    if (size >= 10000) return `${(size / 1000).toFixed(1)}K`;
    return size.toLocaleString("en-IN");
  }
}
