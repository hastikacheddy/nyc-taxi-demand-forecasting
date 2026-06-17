# OPA / Gatekeeper admission policy (Phase V — policy-as-code).
# Deterministically rejects workloads that violate the pod least-privilege
# baseline this project's manifests already follow. Load into an OPA Gatekeeper
# ConstraintTemplate (or `conftest test kubernetes/`) to enforce it in-cluster.
package kubernetes.admission

import future.keywords.in

deny[msg] {
    input.request.kind.kind == "Pod"
    c := input.request.object.spec.containers[_]
    not c.securityContext.runAsNonRoot
    msg := sprintf("container '%v' must set securityContext.runAsNonRoot=true", [c.name])
}

deny[msg] {
    input.request.kind.kind == "Pod"
    c := input.request.object.spec.containers[_]
    not c.securityContext.readOnlyRootFilesystem
    msg := sprintf("container '%v' must set readOnlyRootFilesystem=true", [c.name])
}

deny[msg] {
    input.request.kind.kind == "Pod"
    c := input.request.object.spec.containers[_]
    not "ALL" in c.securityContext.capabilities.drop
    msg := sprintf("container '%v' must drop ALL capabilities", [c.name])
}

deny[msg] {
    input.request.kind.kind == "Pod"
    c := input.request.object.spec.containers[_]
    not c.resources.limits.memory
    msg := sprintf("container '%v' must declare a memory limit", [c.name])
}

deny[msg] {
    input.request.kind.kind == "Pod"
    input.request.object.spec.automountServiceAccountToken == true
    msg := "pods must not auto-mount the service account token"
}
