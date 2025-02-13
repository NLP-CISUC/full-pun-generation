from pathlib import Path

import yaml

credentials_path = Path("config/credentials.yaml")
with credentials_path.open() as f:
    credentials = yaml.safe_load(f)


def authenticate(username, password):
    return username in credentials and credentials[username] == password
