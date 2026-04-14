FROM node:20-alpine

WORKDIR /app

# Copy dependency files first (important for caching)
COPY package*.json ./

# Install dependencies
RUN npm ci --omit=dev || npm install --omit=dev

# Copy full source code
COPY . .

# App port (adjust if needed)
EXPOSE 5000

ENV NODE_ENV=production

# Start server
CMD ["node", "src/server.js"]
