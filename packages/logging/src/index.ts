export const log = (level: 'info' | 'error', message: string, data: Record<string, unknown> = {}) => {
  console.log(JSON.stringify({ level, message, ...data, ts: new Date().toISOString() }));
};
