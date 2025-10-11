#!/bin/bash

# Docker构建和运行脚本
# 使用方法: bash docker_build.sh [build|run|stop|clean]

set -e

IMAGE_NAME="sci2"
CONTAINER_NAME="sci2-container"

case "${1:-build}" in
    "build")
        echo "🔨 构建Docker镜像..."
        docker build -t $IMAGE_NAME .
        echo "✅ 镜像构建完成!"
        ;;
    
    "run")
        echo "🚀 启动Docker容器..."
        # 检查是否已存在容器
        if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
            echo "📦 容器已存在，启动中..."
            docker start $CONTAINER_NAME
            docker exec -it $CONTAINER_NAME /bin/bash
        else
            echo "📦 创建新容器..."
            docker run -it \
                --name $CONTAINER_NAME \
                --gpus all \
                -v $(pwd):/workspace \
                -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
                -e DISPLAY=$DISPLAY \
                -e NVIDIA_VISIBLE_DEVICES=all \
                -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
                $IMAGE_NAME
        fi
        ;;
    
    "stop")
        echo "⏹️  停止Docker容器..."
        docker stop $CONTAINER_NAME
        echo "✅ 容器已停止!"
        ;;
    
    "clean")
        echo "🧹 清理Docker资源..."
        docker stop $CONTAINER_NAME 2>/dev/null || true
        docker rm $CONTAINER_NAME 2>/dev/null || true
        docker rmi $IMAGE_NAME 2>/dev/null || true
        echo "✅ 清理完成!"
        ;;
    
    "logs")
        echo "📋 查看容器日志..."
        docker logs $CONTAINER_NAME
        ;;
    
    *)
        echo "使用方法: $0 [build|run|stop|clean|logs]"
        echo ""
        echo "命令说明:"
        echo "  build  - 构建Docker镜像"
        echo "  run    - 运行Docker容器"
        echo "  stop   - 停止Docker容器"
        echo "  clean  - 清理Docker资源"
        echo "  logs   - 查看容器日志"
        ;;
esac
