# CI/CD

See [Docker](docker.md) and [Kubernetes](kubernetes.md) for the build and deployment steps this chapter automates.

## What is it?

CI/CD (Continuous Integration / Continuous Deployment) is an automated pipeline that runs on every code change: install dependencies, run tests, and, if everything passes, build and deploy the result — replacing a manual "test it, build it, ship it" process with one triggered automatically by a code change.

Docker packages a service, Kubernetes runs and manages it; CI/CD is the automated assembly line connecting a code change to a running deployment, so pushing new code doesn't mean someone has to manually rebuild an image, manually run tests, and manually apply a new Kubernetes manifest every single time.

## Why does it exist?

Every step covered so far in this section — building a Docker image, applying a Kubernetes manifest — is something a person could do by hand. Doing it by hand is slow, easy to get wrong under pressure, and doesn't scale past a small number of changes: a team shipping multiple times a day can't have someone manually rebuild and redeploy for every change, and a manual process is exactly where a rushed step — running the tests, double-checking a config value — gets skipped. CI/CD exists to remove that manual step by triggering an automated pipeline on every code change, consistently, every time, without depending on a person to remember every step.

**What belongs in the automated gate, versus a manual step:** fast, reliable checks — unit tests, linting — belong in CI and should block a deployment if they fail. Fully automatic deployment (continuous deployment) makes sense once the test suite is trusted enough to actually be the gate for production; if the tests can't be fully trusted yet, an explicit manual approval step before deployment is the more honest choice than pretending automation alone is a substitute for that trust.

## How does it work?

A pipeline is triggered by an event, most commonly a push to a repository or a pull request being opened. It runs a series of defined jobs and steps: install dependencies, run tests, build a Docker image (from [Docker](docker.md)), push the image to a registry. A later job can depend on an earlier one succeeding, so a build-and-deploy step never runs if the test job fails — the pipeline stops and reports the failure before anything gets deployed.

## Example

A GitHub Actions workflow verified against current documentation for the house-price service from [Docker](docker.md) and [Kubernetes](kubernetes.md) — tests must pass before the image is even built:

```yaml
name: Test and build house-price-service

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest
      - run: pytest

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v7
      - uses: docker/login-action@v4
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v7
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
```

`build-and-push` declares `needs: test`, which means it never runs at all if the `test` job fails — a broken change gets stopped before an image is ever built, let alone pushed or deployed. In a full deployment pipeline, a third job would apply the new image tag to the Kubernetes Deployment from [Kubernetes](kubernetes.md), again gated on the previous jobs succeeding.

## Where is it used?

Any team shipping code changes more than occasionally, and especially any team running the kind of Docker/Kubernetes deployment covered in the previous two chapters, where manually rebuilding and redeploying on every change doesn't scale.

## Advantages

- **Runs the same checks, the same way, on every single change**, removing the chance of a rushed manual process skipping a step.
- **Catches a broken change before it's ever built into an image or deployed**, rather than after.
- **Makes deployment a routine, repeatable event** instead of a manual, error-prone one.

## Limitations

- **A pipeline is only as good as the tests it runs.** A passing pipeline with weak test coverage gives false confidence, not real assurance the change is actually correct.
- **Fully automatic deployment removes a human checkpoint.** For a high-stakes change, that can be exactly the wrong trade-off without a compensating safeguard — a staged rollout, a manual approval gate.
- **Pipeline configuration itself is code that can have bugs.** A misconfigured pipeline can silently skip a step, or pass when it shouldn't, the same way any other code can be wrong.

## Production considerations

- **Trust in an automated deployment gate should scale with the actual trustworthiness of the test suite behind it.** Deploying automatically on a thin test suite is a real risk, not a convenience.
- **A pipeline's own credentials are a real security surface.** Registry logins and deployment access need the same care as any other production credential, not looser handling just because they live in a CI configuration file.
- **Pipeline failures need to be visible and actionable quickly.** A broken pipeline that nobody notices for days quietly reverts a team back to manual, ad hoc deployment, without anyone deciding that on purpose.

## Common mistakes

- **Treating a green pipeline as proof a change is correct**, rather than proof it passed the specific checks that happen to be automated.
- **Enabling fully automatic deployment before the test suite is actually trustworthy enough** to serve as that gate.
- **Granting a pipeline broader credentials or permissions than the specific job actually needs**, widening the blast radius if the pipeline configuration itself is ever compromised.

## Interview questions

### Basic

- What is the difference between continuous integration and continuous deployment?
- Why does a build-and-push job typically depend on a test job passing first?

### Intermediate

- Why is a passing CI pipeline not the same thing as proof that a change is correct?
- When would a manual approval step before deployment be the right choice over full automation?

### Advanced

- A CI/CD pipeline's credentials are compromised. What about how the pipeline was configured would determine how much damage that causes?
- Design the gating strategy for a deployment pipeline serving a high-stakes production model. What would you automate, and what would you deliberately keep as a manual step?
