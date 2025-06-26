# Step 1: Specify the base OS and Python version.
FROM python:3.12-slim

# Step 2: Install system dependencies (like git) as the root user.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Step 3: Create a new, non-root user and its home directory.
RUN useradd -m -u 1000 user

# Step 4: Switch to the new user.
USER user

# Step 5: Set the PATH environment variable to include the user's local bin directory.
ENV PATH="/home/user/.local/bin:${PATH}"

# Step 6: Set the working directory to a new directory in the user's home.
WORKDIR /home/user/app

# Step 7: Copy the requirements file and install packages as the new user.
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Step 8: Copy the rest of your project files.
COPY --chown=user:user . .

# Step 9: Define the default command to run when the Space starts.
# --- THE FINAL FIX IS HERE ---
# We add `--ServerApp.disable_check_xsrf=True` to tell the server not to
# require the security token that is being blocked by the iframe.
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--port=7860", "--NotebookApp.token=''", "--ServerApp.allow_origin='*'", "--ServerApp.allow_remote_access=True", "--ServerApp.disable_check_xsrf=True"]