docker service create --restart-condition=none --replicas 1 --generic-resource gpu=1 --with-registry-auth --mount type=bind,source=/mnt/fast-scratch/$USER,destination=/scratch -e PYTHONUNBUFFERED=1 -u "$(id -u):1337" --hostname dev-{{.Node.Hostname}} --name $USER-dev-container eidos-service.di.unito.it/$USER/dev:latest