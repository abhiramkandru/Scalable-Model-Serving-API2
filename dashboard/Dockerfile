FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci || npm install
COPY . .
RUN npm run build

# nginx serves the static bundle and proxies /api/* to the serving API.
# The official entrypoint renders /etc/nginx/templates/*.template with the environment, so the upstream is set at run time:
#   docker run -e API_UPSTREAM=gateway:80 ...
FROM nginxinc/nginx-unprivileged:1.27-alpine
ENV API_UPSTREAM=server:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
