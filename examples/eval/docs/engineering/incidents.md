# Incident response

## Severity levels

- **SEV1**: customer data is lost, or the whole fleet stops reporting. Post a status update every
  30 minutes.
- **SEV2**: a single customer's service is degraded. Post a status update every 2 hours.
- **SEV3**: an internal tool is broken, with no customer impact. No status updates required.

## Roles

Every SEV1 and SEV2 incident has an incident commander, who coordinates the response and decides
when it is over. The commander is not the person fixing the problem.

## After the incident

Write a postmortem for every SEV1 and SEV2 incident within five working days. Postmortems are
blameless: they describe what the system allowed to happen, not who made a mistake.
