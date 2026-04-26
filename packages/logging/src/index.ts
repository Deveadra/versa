import { randomUUID } from 'node:crypto';
import { dirname } from 'node:path';
import { appendFile, mkdir } from 'node:fs/promises';
import { TelemetryEventSchema, type TraceContext, type TelemetryEvent } from '@versa/shared';

type LegacyLogLevel = 'info' | 'error';
type TelemetryLevel = 'debug' | 'info' | 'warn' | 'error';

type TelemetrySink = (event: TelemetryEvent) => void;
export type { TraceContext };

type SinkOptions = {
  consoleEnabled?: boolean;
  sinks?: TelemetrySink[];
};

type TelemetryActor = {
  service: string;
  source: string;
  actorId?: string;
};

type LoggerOptions = {
  actor: TelemetryActor;
  context?: Partial<TraceContext>;
  sink?: TelemetrySink;
};

type TelemetryInput = {
  eventType: string;
  message: string;
  level?: TelemetryLevel;
  context?: Partial<TraceContext>;
  attributes?: Record<string, unknown>;
};

type RequestLike = {
  method?: string;
  originalUrl?: string;
  path?: string;
  headers?: Record<string, string | string[] | undefined>;
  telemetryContext?: Partial<TraceContext>;
};

type ResponseLike = {
  statusCode: number;
  on: (event: 'finish', listener: () => void) => void;
};

type NextLike = () => void;

const readHeaderValue = (
  headers: Record<string, string | string[] | undefined> | undefined,
  key: string,
) => {
  const value = headers?.[key.toLowerCase()] ?? headers?.[key];
  if (Array.isArray(value)) {
    const first = value[0]?.trim();
    return first ? first : undefined;
  }

  if (typeof value === 'string') {
    const normalized = value.trim();
    return normalized.length > 0 ? normalized : undefined;
  }

  return undefined;
};

const ensureTraceContext = (context: Partial<TraceContext> = {}): TraceContext => ({
  traceId: context.traceId ?? randomUUID(),
  correlationId: context.correlationId,
  runId: context.runId,
  requestId: context.requestId,
  parentTraceId: context.parentTraceId,
});

const consoleSink: TelemetrySink = (event) => {
  console.log(JSON.stringify(event));
};

export const createNdjsonFileSink = (path: string): TelemetrySink => {
  let sinkDisabled = false;
  let sinkErrorReported = false;

  const ensureDirectoryReady = mkdir(dirname(path), { recursive: true })
    .then(() => true)
    .catch((error) => {
      sinkDisabled = true;

      if (!sinkErrorReported) {
        sinkErrorReported = true;
        console.error(
          JSON.stringify({
            level: 'error',
            eventType: 'telemetry.sink.error',
            message: 'failed to initialize telemetry sink directory',
            path,
            error: (error as Error).message,
            timestamp: new Date().toISOString(),
          }),
        );
      }

      return false;
    });

  return (event) => {
    if (sinkDisabled) return;

    void ensureDirectoryReady
      .then((ready) => (ready ? appendFile(path, `${JSON.stringify(event)}\n`) : undefined))
      .catch((error) => {
        sinkDisabled = true;

        if (!sinkErrorReported) {
          sinkErrorReported = true;
          console.error(
            JSON.stringify({
              level: 'error',
              eventType: 'telemetry.sink.error',
              message: 'failed to write telemetry event',
              path,
              error: (error as Error).message,
              timestamp: new Date().toISOString(),
            }),
          );
        }
      });
  };
};

export const createTelemetrySink = ({ consoleEnabled = true, sinks = [] }: SinkOptions): TelemetrySink => {
  return (event) => {
    if (consoleEnabled) {
      consoleSink(event);
    }

    sinks.forEach((sink) => sink(event));
  };
};

export type TelemetryRequest = RequestLike;

export const getTelemetryContext = (req: TelemetryRequest): Partial<TraceContext> =>
  req.telemetryContext ?? {};

export const createLogger = ({ actor, context = {}, sink = consoleSink }: LoggerOptions) => {
  const emit = ({ eventType, message, level = 'info', context: eventContext, attributes = {} }: TelemetryInput) => {
    const event = TelemetryEventSchema.parse({
      eventId: `evt_${randomUUID().slice(0, 8)}`,
      eventType,
      level,
      message,
      timestamp: new Date().toISOString(),
      actor,
      context: ensureTraceContext({ ...context, ...eventContext }),
      attributes,
    });

    sink(event);
    return event;
  };

  return {
    emit,
    debug: (eventType: string, message: string, attributes: Record<string, unknown> = {}, eventContext = {}) =>
      emit({ eventType, message, level: 'debug', attributes, context: eventContext }),
    info: (eventType: string, message: string, attributes: Record<string, unknown> = {}, eventContext = {}) =>
      emit({ eventType, message, level: 'info', attributes, context: eventContext }),
    warn: (eventType: string, message: string, attributes: Record<string, unknown> = {}, eventContext = {}) =>
      emit({ eventType, message, level: 'warn', attributes, context: eventContext }),
    error: (eventType: string, message: string, attributes: Record<string, unknown> = {}, eventContext = {}) =>
      emit({ eventType, message, level: 'error', attributes, context: eventContext }),
    child: (overrides: Partial<LoggerOptions> & { context?: Partial<TraceContext> } = {}) =>
      createLogger({
        actor: { ...actor, ...overrides.actor },
        context: { ...context, ...overrides.context },
        sink: overrides.sink ?? sink,
      }),
  };
};

export const createRequestTelemetryMiddleware = (logger: ReturnType<typeof createLogger>) => {
  return (req: RequestLike, res: ResponseLike, next: NextLike) => {
    const startedAt = Date.now();
    const context = ensureTraceContext({
      traceId: readHeaderValue(req.headers, 'x-trace-id'),
      correlationId: readHeaderValue(req.headers, 'x-correlation-id'),
      runId: readHeaderValue(req.headers, 'x-run-id'),
      requestId: readHeaderValue(req.headers, 'x-request-id') ?? randomUUID(),
    });

    req.telemetryContext = context;

    logger.info(
      'http.request.started',
      'http request started',
      {
        method: req.method,
        path: req.originalUrl ?? req.path,
      },
      context,
    );

    res.on('finish', () => {
      logger.info(
        'http.request.completed',
        'http request completed',
        {
          method: req.method,
          path: req.originalUrl ?? req.path,
          statusCode: res.statusCode,
          durationMs: Date.now() - startedAt,
        },
        context,
      );
    });

    next();
  };
};

const legacyLogger = createLogger({
  actor: {
    service: 'legacy-runtime',
    source: 'legacy-log-api',
  },
});

export const log = (
  level: LegacyLogLevel,
  message: string,
  data: Record<string, unknown> = {},
) => {
  legacyLogger.emit({
    eventType: message,
    message,
    level,
    attributes: data,
  });
};
