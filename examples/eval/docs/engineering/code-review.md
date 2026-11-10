# Code review

Every change is merged through a pull request.

## Approvals

Firmware changes need two approvals. Changes to the web dashboard and internal tools need one.

## Keep it small

Aim for pull requests under 400 changed lines. Large changes are harder to review and more
likely to hide bugs; split them into a series.

## Expectations

Reviewers respond within one working day, even if only to say when they will review. The
continuous integration build must pass before merging. Pull requests are squash-merged so each
one becomes a single commit on the main branch.
