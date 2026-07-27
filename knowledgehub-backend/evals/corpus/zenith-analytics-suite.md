# Zenith Analytics Suite — Product Overview

## What Zenith Analytics Suite is

Zenith Analytics Suite is a self-hosted business intelligence tool. It connects to
existing data warehouses and lets analysts build dashboards without writing SQL.

## Services

Zenith Analytics Suite ships as two components:

- **Zenith Explore** — a drag-and-drop dashboard builder that queries the connected
  warehouse directly. It supports Snowflake, BigQuery, and Postgres.
- **Zenith Pipeline** — a scheduled transformation runner that materialises
  intermediate tables on a cron schedule.

## Pricing

Zenith Analytics Suite is licensed per seat, not per query.

- Zenith Explore costs $45 per analyst seat per month. Viewer-only accounts are free
  and unlimited.
- Zenith Pipeline costs a flat $600 per month regardless of how many jobs run.

There is no free tier. A 30-day trial is available with all features unlocked.
Non-profit organisations receive a 40% discount on all seats.

## Deployment

Zenith Analytics Suite is self-hosted only — there is no vendor-managed cloud option.
It is distributed as a Docker Compose bundle and requires at minimum 4 vCPUs and
16 GiB of memory.

## Support

Support is included in the licence cost. The support team answers within one
business day. There is no paid upgrade tier and no phone support.
