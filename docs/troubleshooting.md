# Troubleshooting

- If SQLite file lock occurs, stop running servers and rerun `pnpm db:reset`.
- If web cannot fetch tasks, verify core is running on `:4000`.
