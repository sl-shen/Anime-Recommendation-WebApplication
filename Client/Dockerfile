# 构建阶段
FROM node:16 as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
RUN ls -la  # 调试：列出当前目录内容
RUN ls -la dist || echo "dist directory not found"  # 调试：列出 dist 目录内容（如果存在）

# 生产阶段
FROM node:16-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=build /app/dist ./dist
RUN ls -la  # 调试：列出当前目录内容
EXPOSE 7979
CMD ["serve", "-s", "dist", "-l", "7979"]