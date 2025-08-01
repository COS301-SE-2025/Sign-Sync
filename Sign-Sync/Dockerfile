#base image
FROM node:20

#set working directory
WORKDIR /app

#copy package files and install dependencies
COPY package*.json ./
RUN npm install

#copy the rest of the app
COPY . .

#build the project (frontend + transpile backend)
RUN npm run build

#set environment variable to production
ENV NODE_ENV=production

#expose the port defined in .env (fallback to 3000)
ARG PORT=3000
ENV PORT=${PORT}
EXPOSE ${PORT}

#start the backend server
CMD ["node", "backend/dist/server.js"]