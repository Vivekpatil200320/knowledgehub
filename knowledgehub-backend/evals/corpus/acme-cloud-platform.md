# Acme Cloud Platform — Product Overview

## What Acme Cloud Platform is

Acme Cloud Platform is a managed container hosting service. It runs containerised
workloads across three regions (us-east, eu-west, ap-south) without requiring
customers to manage Kubernetes clusters themselves.

## Services

Acme Cloud Platform offers three services:

- **Acme Run** — deploys a container image and scales it automatically based on
  incoming request volume. Cold start latency is typically under 400ms.
- **Acme Queue** — a managed message queue with at-least-once delivery guarantees
  and a maximum message retention window of 14 days.
- **Acme Vault** — encrypted secret storage with automatic key rotation every
  90 days.

## Pricing

Acme Cloud Platform is billed monthly on a per-service basis.

- Acme Run costs $0.000024 per vCPU-second plus $0.0000025 per GiB-second of memory.
  The free tier includes 180,000 vCPU-seconds per month.
- Acme Queue costs $0.40 per million messages published. The first million messages
  each month are free.
- Acme Vault costs $0.03 per secret per month, with no free tier.

Annual prepayment reduces the total bill by 15%. There are no charges for data
transfer between services inside the same region.

## Support tiers

Basic support is included at no cost and answers within two business days.
Standard support costs $99 per month and answers within four business hours.
Enterprise support costs $2,500 per month and provides a dedicated technical
account manager with a 30-minute response target.

## Service level agreement

Acme Run guarantees 99.95% monthly uptime. Acme Queue and Acme Vault each
guarantee 99.9% monthly uptime. If uptime falls below the guarantee, customers
receive service credits equal to 10% of that service's monthly bill.
