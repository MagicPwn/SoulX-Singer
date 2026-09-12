# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# Serve stage - Hugging Face Spaces use port 7860
FROM nginx:alpine

# Custom nginx config for SPA and port 7860
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 7860

CMD ["nginx", "-g", "daemon off;"]
