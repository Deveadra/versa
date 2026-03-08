# AI Adapter Service

This service defines the boundary for Ultron/Aerith capabilities.
Core calls this adapter with timeout/retry (future middleware); if unreachable, core continues without AI.
