/** @type {import('next').NextConfig} */
const nextConfig = {
  // output: "standalone",  // Only enable for production Docker builds — causes high CPU in dev
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["recharts", "lightweight-charts"],
  },
  // Reduce memory: evict compiled pages faster in dev
  onDemandEntries: {
    maxInactiveAge: 60 * 1000,   // 60s before evicting inactive page
    pagesBufferLength: 3,         // only keep 3 pages compiled
  },
  webpack: (config, { dev }) => {
    // Reduce dev server CPU: poll less aggressively
    config.watchOptions = {
      ...config.watchOptions,
      poll: 3000,
      aggregateTimeout: 1000,
      ignored: ["**/node_modules/**", "**/.next/**", "**/.git/**"],
    };
    // In dev, disable CSS content hash to prevent HMR loops from hash changes
    if (dev) {
      const miniCss = config.plugins?.find(
        (p) => p.constructor.name === "CssMinimizerPlugin"
      );
      if (miniCss) miniCss.options = { ...miniCss.options, filename: "[name].css" };
    }
    return config;
  },
};

export default nextConfig;
