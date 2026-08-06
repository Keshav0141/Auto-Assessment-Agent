FROM python:3.11-slim

WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code

# Set environment variables for Hugging Face Spaces
ENV HOST=0.0.0.0
ENV PORT=7860

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
