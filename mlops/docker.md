# Docker

See [FastAPI](fastapi.md) for the service this chapter packages and runs.

## What is it?

Docker is a tool for packaging an application together with everything it needs to run — code, dependencies, runtime — into a single, portable unit called a **container**, so it runs the same way regardless of where it's deployed.

If FastAPI defines the interface a model is served through, Docker defines the exact environment that service runs inside — the same Python version, the same installed package versions, the same OS-level dependencies, packaged together so "it works on my machine" and "it works on the production server" mean the same thing.

## Why does it exist?

A Python service depends on more than just its own code: a specific Python version, specific package versions — exactly the kind of environment mismatch already seen with pandas' Copy-on-Write default changing between versions in [Pandas](../python-for-ml/pandas.md) — and sometimes OS-level libraries. Before containers, deploying a service meant carefully replicating that entire environment by hand on every machine it needed to run on, a process prone to drift, where a production server's environment quietly differs from a developer's machine and causes bugs that only show up in one place. Docker exists to eliminate that drift: the exact environment is defined once, in a file, and built into an image that runs identically anywhere Docker itself is installed.

**When a service needs containerizing, and when it doesn't:** almost any service that will run somewhere other than the machine that built it benefits from containerization, since it removes an entire class of "works here, not there" bugs. It's most valuable specifically when a service has real dependencies pinned to exact versions — a specific scikit-learn or PyTorch version, for instance. A genuinely dependency-free script running on a single, tightly controlled machine might not need it, but that's the exception for a real production model-serving service, not the norm.

## How does it work?

A **Dockerfile** describes, step by step, how to build the image: which base image to start from, which files to copy in, which dependencies to install, and what command to run when a container starts. `docker build` reads the Dockerfile and produces an **image** — a snapshot of that environment. `docker run` starts a **container** from that image — a running instance of it, isolated from the host machine except where explicitly connected, such as a specific network port.

## Example

Packaging the [FastAPI](fastapi.md) house-price service into a real image and running it — built and run for real to verify this chapter, then removed afterward:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t house-price-service .
docker run -d --name house-price-service -p 8000:8000 house-price-service
```

The running container's app returned exactly the same prediction already verified in [FastAPI](fastapi.md):

```text
{"predicted_price_thousands": 253.97727272727272}
```

The `python:3.11-slim` base image, the pinned package versions in `requirements.txt`, and the app code are all packaged together into one image — the same image produces the same result on any machine with Docker installed, regardless of what Python or package versions happen to already be on that machine.

## Where is it used?

Packaging a model-serving API (like the one from [FastAPI](fastapi.md)) for deployment, running the exact same environment in development, staging, and production, and as the standard unit of deployment for the orchestration covered in the next chapter, Kubernetes.

## Advantages

- **Eliminates environment drift** between where a service is built and where it runs — the same image runs identically everywhere Docker is installed.
- **Packages dependencies at exact, pinned versions**, avoiding the kind of version-mismatch surprises already seen directly in [Pandas](../python-for-ml/pandas.md).
- **Makes a service's runtime environment reproducible and inspectable**, defined in a single file rather than assembled by hand.

## Limitations

- **Doesn't fix a bug in the application code itself.** A model that's wrong, or an API that's misconfigured, behaves exactly as wrong inside a container as outside one.
- **Adds a real build step and image size to manage.** A bloated image — unnecessary dependencies, unpinned versions pulling in more than expected — slows down builds and deployments.
- **Isolation is not the same as security by default.** A container still needs to be configured deliberately to limit what it can access on the host or network.

## Production considerations

- **Image size and build time directly affect deployment speed.** A large, unoptimized image makes every deploy slower and more expensive to store and transfer.
- **Pinning exact dependency versions is what actually prevents the "it worked yesterday" class of bug.** An unpinned dependency can silently change behavior on a rebuild, weeks or months after the original image was built.
- **A running container is not automatically monitored, restarted on failure, or scaled.** Those are exactly the responsibilities the next chapter, Kubernetes, exists to hand off to.

## Common mistakes

- **Not pinning dependency versions in the image**, defeating the entire point of eliminating environment drift — a rebuild months later can silently pull in different package versions than the original.
- **Treating a large, slow-to-build image as an acceptable cost**, rather than an ongoing tax on every deployment and every developer waiting for a build.
- **Assuming a container running successfully means the service is production-ready**, when monitoring, restart behavior, and scaling still need to be addressed separately.

## Interview questions

### Basic

- What problem does Docker solve that installing dependencies directly on a server doesn't?
- What is the difference between a Docker image and a Docker container?

### Intermediate

- Why does pinning exact dependency versions in a Dockerfile matter, rather than just listing package names?
- What does containerizing a service not solve on its own?

### Advanced

- A containerized model-serving API works correctly in a build but starts failing in production weeks later, with no code changes. What would you check first, given what a Dockerfile actually pins and doesn't pin?
- Why is a running container not sufficient on its own for a production deployment, and what responsibilities still need to be handled separately?
