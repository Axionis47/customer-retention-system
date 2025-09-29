"""E2E Docker smoke tests."""
import subprocess
import time

import pytest
import requests


@pytest.mark.e2e
@pytest.mark.slow
def test_docker_app_smoke():
    """Test that app Docker image builds and runs."""
    # Build image
    print("Building Docker image...")
    build_result = subprocess.run(
        ["docker", "build", "-f", "ops/docker/Dockerfile.app", "-t", "churn-saver-app:test", "."],
        capture_output=True,
        text=True,
    )

    if build_result.returncode != 0:
        pytest.skip(f"Docker build failed: {build_result.stderr}")

    # Run container
    print("Starting container...")
    run_result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "churn-saver-test",
            "-p",
            "8080:8080",
            "churn-saver-app:test",
        ],
        capture_output=True,
        text=True,
    )

    if run_result.returncode != 0:
        pytest.skip(f"Docker run failed: {run_result.stderr}")

    container_id = run_result.stdout.strip()

    try:
        # Wait for startup
        print("Waiting for service to start...")
        time.sleep(5)

        # Test health endpoint
        print("Testing /healthz...")
        response = requests.get("http://localhost:8080/healthz", timeout=5)
        assert response.status_code == 200, f"Health check failed: {response.text}"

        # Test retain endpoint
        print("Testing /retain...")
        payload = {
            "customer_id": "C12345",
            "churn_risk": 0.75,
            "tenure_months": 24,
            "monthly_spend": 89.99,
        }

        response = requests.post("http://localhost:8080/retain", json=payload, timeout=10)
        assert response.status_code == 200, f"Retain endpoint failed: {response.text}"

        data = response.json()
        assert "decision" in data, "Response should contain decision"
        assert "customer_id" in data, "Response should contain customer_id"

        print("✓ Docker smoke test passed")

    except Exception as e:
        # Print container logs on failure
        logs_result = subprocess.run(
            ["docker", "logs", container_id],
            capture_output=True,
            text=True,
        )
        print(f"Container logs:\n{logs_result.stdout}")
        raise e

    finally:
        # Cleanup
        print("Cleaning up container...")
        subprocess.run(["docker", "stop", container_id], capture_output=True)
        subprocess.run(["docker", "rm", container_id], capture_output=True)


@pytest.mark.e2e
def test_docker_build_only():
    """Test that Docker images build successfully (no run)."""
    # Test app image
    print("Building app image...")
    result = subprocess.run(
        ["docker", "build", "-f", "ops/docker/Dockerfile.app", "-t", "churn-saver-app:test", "."],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Build output:\n{result.stdout}")
        print(f"Build errors:\n{result.stderr}")

    assert result.returncode == 0, "App Docker build should succeed"

    # Test trainer image
    print("Building trainer image...")
    result = subprocess.run(
        ["docker", "build", "-f", "ops/docker/Dockerfile.trainer", "-t", "churn-saver-trainer:test", "."],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Build output:\n{result.stdout}")
        print(f"Build errors:\n{result.stderr}")

    assert result.returncode == 0, "Trainer Docker build should succeed"

