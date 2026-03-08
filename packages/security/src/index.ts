export type SecretRef = { key: string; value: string };

export const redact = (input: string) => input.replace(/(token|secret|password)=\S+/gi, '$1=[REDACTED]');

export const sensitivityClasses = ['public', 'internal', 'private', 'restricted'] as const;
