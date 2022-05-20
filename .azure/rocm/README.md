# Azure DevOps ROCm Build Agent

## Building

On a Linux machine with Docker installed:

```shell
sh build_agent.sh
```

After building, you can manually push the container to a Docker repository if needed.

## Running

On a Linux machine with an AMD GPU:

```shell
export AZP_URL=<ADO org-level server url>
export AZP_TOKEN=<ADO server PAT>
export ACR_USER=<ACR client id>
export ACR_SECRET=<ACR client secret>
bash run_agent.sh
```

Where:
- <PAT> - Personal access token with "Agent Pools (read, manage)" scope.
- <ADO_URL> - Server URL for the Azure DevOps instance. Note that this is the organization-level URL, *not* the project-level URL. This is likely because ADO agents and pools can be organization-scoped.
- <ACR_USER> - Client id for the service principal allowing pull access to the Azure container registry
- <ACR_SECRET> - Client secret for the service principal allowing pull access to the Azure container registry

## Debugging

See the example debugging code in `run_agent.sh` to run the container interactively. By default the scripts directory will be mounted as /azp, so you can make changes to `start.sh` without rebuilding the container.

Note that running the container interactively means it'll just launch a bash shell into the container so that you can try stuff. It won't start the agent unless you run `start.sh` manually.

## Stopping

Stopping is a two-step process because the container is configured to restart when stopped:
1. Ensure no active jobs are running
2. Find the container name: `sudo docker ps`
3. `sudo docker kill <container_name>`
4. From the Web UI, browse to your Agent Pool and delete the agent from the pool
5. You may need to repeat steps 3 and 4 if the container does not exit immediately

## References
 - https://docs.microsoft.com/en-us/azure/devops/pipelines/agents/docker?view=azure-devops
 - https://docs.microsoft.com/en-us/azure/container-registry/container-registry-auth-service-principal