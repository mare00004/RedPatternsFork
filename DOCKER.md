# Docker

The Important files are

 - `Dockerfile`. Tells docker how the container should look
 - `.dockerignore`. Tells docker what files to ignore, which is imporant so you don't accidentally include the build directory, which confuses CMake.

To build the container run

`sudo docker build -t <name>:<tag> .`

If you want to push the container to docker hub, then you have to log into your account with `sudo docker -u <user> -p <PAT>` (PAT = Personal Access Token), make sure the `<name>` matches your repository name on dockerhub and push the container with `sudo docker push <name>:<tag>`.

It is imporant that you add a unique `<tag>` to each container version, so that you force HTCondor to pull the new version of your container instead of a cached version.

As a tag you could use the current git-hash by replacing `<tag>` with `git rev-parse --short HEAD`

