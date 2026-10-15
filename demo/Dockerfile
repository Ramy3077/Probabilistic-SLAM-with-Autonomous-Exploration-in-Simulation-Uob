# Static hosting for the demo (used by Railway). Build context: repo root.
FROM alpine:3.20 AS build
WORKDIR /src
COPY controllers/robot_controller controllers/robot_controller
COPY demo demo
RUN sh demo/build.sh

FROM caddy:2.8-alpine
COPY demo/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /src/demo/dist /srv
