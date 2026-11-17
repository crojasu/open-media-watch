# Security policy

Open Media Watch is an early-stage project maintained by one person.

## Reporting a vulnerability

Please report security problems privately, not in a public issue: use GitHub's "Report a vulnerability" button (Security tab of this repository) or write to **rojas.catalina@gmail.com**.
Include what you found, how to reproduce it, and which version or commit you tested.
You will get a reply as soon as the maintainer can, with no guaranteed response time.

## Scope notes

- The code ships with no credentials. Configure secrets through environment variables (see `.env.example`) and never commit a `.env` file.
- The write API for dossiers uses HTTP Basic credentials from `REVIEW_USERS`; run it behind HTTPS.
