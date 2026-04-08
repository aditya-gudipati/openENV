FROM python:3.11-slim

# Prepare Hugging Face Spaces specific environment limitations
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy deployment requirements securely matched to User
COPY --chown=user requirements.txt .

# Install dependencies deterministically
RUN pip install --no-cache-dir -r requirements.txt

# Copy source configuration
COPY --chown=user *.py /app/
COPY --chown=user server/ /app/server/
COPY --chown=user openenv.yaml /app/openenv.yaml

# Expose validation mapping port strictly assigned universally by HF Spaces
EXPOSE 7860

# Stand up the HTTP layer natively on Space port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
