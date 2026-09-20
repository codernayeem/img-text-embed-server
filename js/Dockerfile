FROM node:20-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package files and install dependencies
COPY package.json ./
RUN npm install --omit=dev

# Copy application files
COPY . .

# Expose server port
EXPOSE 8001

# Run Node.js server
CMD ["node", "server.js"]
