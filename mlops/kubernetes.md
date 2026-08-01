# Kubernetes

See [Docker](docker.md) for the container this chapter runs and manages at scale.

## What is it?

Kubernetes is a system for running and managing containers across a cluster of machines — deciding where each container runs, restarting it if it fails, routing traffic to it, and scaling the number of running copies up or down.

Docker packages a service into a container and can run one. Kubernetes is the layer that runs many containers across many machines, watches them, and keeps the actual state of the cluster matching what was declared it should be — if a container dies, Kubernetes notices and starts a replacement, without anyone needing to watch it manually.

## Why does it exist?

[Docker](docker.md#production-considerations) named this gap directly: "a running container is not automatically monitored, restarted on failure, or scaled." Kubernetes exists to take on exactly those responsibilities at scale. For a single container on a single machine, a person, or a simple script, could restart it manually if it crashed. For a real production service — multiple replicas, spread across multiple machines, needing to handle machine failures, traffic spikes, and rolling updates without downtime — that manual approach doesn't scale. Kubernetes exists to automate it: describe the desired state (how many replicas, what image, what resources), and Kubernetes continuously works to keep the actual cluster state matching that description.

**When a service needs Kubernetes, and when Docker alone is enough:** a single service on a single machine, with modest traffic and tolerance for brief manual intervention if it fails, doesn't need Kubernetes — Docker alone, with a simple restart policy, is enough. Kubernetes earns its real complexity specifically when a service needs multiple replicas for reliability or scale, needs to run across multiple machines, or needs automated rollout and rollback of new versions without downtime — capabilities that would otherwise need to be built and maintained by hand.

## How does it work?

A **Deployment** describes the desired state of a set of identical containers: which image to run, how many replicas, and how updates should roll out. Kubernetes continuously compares the actual running state to the Deployment's desired state, starting new containers (Pods) if there are too few, and removing extras if there are too many. A **Service** gives a stable network address to a set of Pods, so other parts of the system can reach them without needing to track individual Pods' addresses as they're created and destroyed. **Readiness and liveness probes** let Kubernetes check whether a container is actually healthy — a failing liveness probe gets the container restarted; a failing readiness probe temporarily removes it from receiving traffic, without a restart.

## Example

The YAML below is verified against current Kubernetes documentation, not run against a live cluster — a deliberate choice for this chapter, unlike [Docker](docker.md)'s example, which was actually built and run. A Deployment and Service for the house-price image from [Docker](docker.md):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: house-price-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: house-price-service
  template:
    metadata:
      labels:
        app: house-price-service
    spec:
      containers:
      - name: house-price-service
        image: house-price-service:latest
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /docs
            port: 8000
          initialDelaySeconds: 3
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /docs
            port: 8000
          initialDelaySeconds: 3
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: house-price-service
spec:
  selector:
    app: house-price-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

`replicas: 3` asks Kubernetes to keep exactly three copies of the container running at all times — if one crashes, or the machine it's on fails, Kubernetes starts a replacement automatically, without anyone paging on-call to restart it by hand. The Service gives those three replicas one stable address other services can send requests to; Kubernetes handles routing a given request to whichever replica is actually ready.

## Where is it used?

Running production services that need multiple replicas for reliability, need to scale up and down with demand, or need rolling updates without downtime — a common target for deploying the kind of containerized model-serving API built in [FastAPI](fastapi.md) and [Docker](docker.md).

## Advantages

- **Automatically restarts failed containers and replaces failed nodes**, without manual intervention.
- **Scales the number of running replicas up or down to match demand**, and rolls out new versions gradually rather than all at once.
- **Gives a stable network identity to a set of replicas** that are individually being created and destroyed over time.

## Limitations

- **A real, meaningful operational learning curve.** Running a cluster well requires understanding concepts — Pods, Deployments, Services, and more — that a single-container Docker setup never requires.
- **Doesn't fix an application problem.** A crashing container gets restarted, repeatedly, but Kubernetes has no way to know or fix why it's crashing in the first place.
- **Overkill for a small, low-traffic service** that a single container, or a small number of them, would handle just fine — the operational overhead of running a cluster isn't free.

## Production considerations

- **Readiness and liveness probes need to reflect real health, not just "the process is running."** A probe that always returns success defeats the entire point of automated health checking.
- **Rolling updates change how many old and new replicas coexist during a deployment.** A service that can't tolerate two different versions running simultaneously needs a different rollout strategy, not the default one.
- **Resource limits (CPU, memory) per container decide how many replicas actually fit on a given machine.** An unset or wrong limit can starve other services on the same node, or get a container killed unexpectedly for exceeding memory it was never told about.

## Common mistakes

- **Writing a readiness or liveness probe that doesn't actually check anything meaningful**, so a genuinely broken container still looks "healthy" to Kubernetes.
- **Assuming a container being restarted repeatedly counts as "handling" the problem**, rather than investigating why it's crashing in the first place.
- **Deploying a small, low-traffic service onto a full Kubernetes setup by default**, taking on real operational complexity the service's actual scale doesn't need.

## Interview questions

### Basic

- What problem does Kubernetes solve that Docker alone doesn't?
- What is the difference between a Deployment and a Service?

### Intermediate

- What's the difference between a readiness probe and a liveness probe, and what does each failure actually cause?
- Why might a small, low-traffic service not need Kubernetes at all?

### Advanced

- A container is being restarted repeatedly by Kubernetes. What would you investigate, and why doesn't the restart itself count as fixing the problem?
- Design a rollout strategy for a service that cannot tolerate two different versions running simultaneously. What would break with Kubernetes' default rolling update, and how would you address it?
