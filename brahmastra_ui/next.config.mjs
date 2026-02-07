/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ["recharts", "lightweight-charts"],
  },
};

export default nextConfig;
