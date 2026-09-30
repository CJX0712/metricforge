# 作者: 晨星
# MetricForge —— CPU-only 复现镜像
FROM python:3.13-slim

WORKDIR /app

# 系统依赖（numpy/scipy 需要编译工具链，slim 版需补充）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest ruff

COPY . .

# 健康检查：端到端 demo 能跑通并落盘 benchmark.json
CMD ["python", "examples/run_demo.py"]
