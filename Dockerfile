FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create __init__ files
RUN touch core/__init__.py tasks/__init__.py data/__init__.py evaluation/__init__.py

# Expose both task ports
EXPOSE 8000 8001

# Default: run Task A
# Override with: docker run -e TASK=B ...
ENV TASK=A
ENV ANTHROPIC_API_KEY=""

CMD ["sh", "-c", \
  "if [ \"$TASK\" = 'B' ]; then \
     uvicorn tasks.task_b:create_app --factory --host 0.0.0.0 --port 8001; \
   else \
     uvicorn tasks.task_a:create_app --factory --host 0.0.0.0 --port 8000; \
   fi"]
