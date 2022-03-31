from app import app 
import pytest
from starlette.testclient import TestClient
from utils.all_utills import read_yaml
import os

@pytest.fixture(scope="module")
def test_app():
    client = TestClient(app)
    yield client


@pytest.fixture(scope="module")
def test_read_yaml():
    data = read_yaml(os.path.join("config","config.yaml"))
    return data




