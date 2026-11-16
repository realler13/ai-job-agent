import type { NextConfig } from 'next'
const nextConfig: NextConfig = {
  experimental: {
    // The /api rewrite proxies to FastAPI, whose CV / cover-letter / company-
    // intel endpoints are synchronous and routinely run past Next's 30s default
    // proxy timeout: a full generation is the write plus a truth judge, a
    // hiring-manager grader, the length gate and a cover letter. The browser
    // got "socket hang up" while the server finished and saved the files anyway.
    // Sized from a measured run at the xhigh generation pin (2026-09-26): CV
    // 125s + cover letter 79s = 204s before the judges, so one full
    // generate-docs with a revision pass can reach ~6 minutes. 15 minutes
    // covers a single job with margin. batch-generate runs as a background job
    // (POST returns at once, the Kanban polls its status), so it never waits here.
    proxyTimeout: 900_000,
  },
  async rewrites() {
    return [{ source: '/api/:path*', destination: 'http://localhost:8502/api/:path*' }]
  }
}
export default nextConfig
