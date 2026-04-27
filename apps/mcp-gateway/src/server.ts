import express, { Request, Response } from 'express';
import { loadConfig } from '@versa/config';
import {
  createLogger,
  createNdjsonFileSink,
  createRequestTelemetryMiddleware,
  createTelemetrySink,
  getTelemetryContext,
} from '@versa/logging';
import { buildGatewayHealth, listCapabilities, lookupCapability } from './registry';

const app = express();
const cfg = loadConfig();

const telemetryFileSink = createNdjsonFileSink('artifacts/telemetry.ndjson');
const logger = createLogger({
  actor: {
    service: 'mcp-gateway',
    source: 'http',
  },
  sink: createTelemetrySink({
    consoleEnabled: cfg.TELEMETRY_CONSOLE_ENABLED,
    sinks: cfg.TELEMETRY_ENABLED ? [telemetryFileSink] : [],
  }),
});

const registration = listCapabilities();

app.use(express.json());
app.use(createRequestTelemetryMiddleware(logger));

const buildHealth = () => buildGatewayHealth(cfg, process.uptime() * 1000);
const healthResponse = () => ({ ok: true, data: buildHealth() });

app.get('/health', (_req: Request, res: Response) => {
  res.json(healthResponse());
});

app.get('/mcp/health', (_req: Request, res: Response) => {
  res.json(healthResponse());
});

app.get('/mcp/capabilities', (req: Request, res: Response) => {
  const context = getTelemetryContext(req);
  logger.info(
    'mcp.registry.listed',
      'mcp capability registry listed',
      {
      capabilityCount: registration.count,
      },
    context,
  );

  res.json({ data: registration });
});

app.get('/mcp/capabilities/:capabilityId', (req: Request, res: Response) => {
  const capabilityId = String(req.params.capabilityId);
  const result = lookupCapability(capabilityId);

  if (!result.found) {
    return res.status(404).json({ data: result });
  }

  return res.json({ data: result });
});

const shouldStartHttpListener = cfg.MCP_TRANSPORT === 'http';

const server = shouldStartHttpListener
  ? app.listen(cfg.MCP_GATEWAY_PORT, () => {
      logger.info('app.started', 'mcp gateway server started', {
        port: cfg.MCP_GATEWAY_PORT,
        transport: cfg.MCP_TRANSPORT,
        mcpEnabled: cfg.MCP_ENABLED,
      });
    })
  : null;

if (!shouldStartHttpListener) {
  logger.info('app.started', 'mcp gateway started in stdio mode without http listener', {
    transport: cfg.MCP_TRANSPORT,
    mcpEnabled: cfg.MCP_ENABLED,
  });
}

const handleShutdown = (signal: string) => {
  logger.info('app.shutdown.requested', 'mcp gateway shutdown requested', { signal });
  if (!server) {
    logger.info('app.stopped', 'mcp gateway stopped', { signal });
    return;
  }

  server.close(() => {
    logger.info('app.stopped', 'mcp gateway stopped', { signal });
  });
};

process.once('SIGINT', () => handleShutdown('SIGINT'));
process.once('SIGTERM', () => handleShutdown('SIGTERM'));
