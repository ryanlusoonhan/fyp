import pathlib
import unittest

import yaml


class TestContainerization(unittest.TestCase):
    def test_docker_compose_has_app_service_and_port_mapping(self):
        compose_file = pathlib.Path("docker-compose.yml")
        self.assertTrue(compose_file.exists(), "docker-compose.yml should exist at project root.")

        compose = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
        self.assertIn("services", compose)
        self.assertIn("app", compose["services"])

        app_service = compose["services"]["app"]
        self.assertIn("ports", app_service)
        self.assertIn("3000:3000", app_service["ports"])

    def test_dockerfile_and_start_script_exist(self):
        dockerfile = pathlib.Path("docker/Dockerfile")
        start_script = pathlib.Path("scripts/docker/start-web.sh")

        self.assertTrue(dockerfile.exists(), "docker/Dockerfile should exist.")
        self.assertTrue(start_script.exists(), "scripts/docker/start-web.sh should exist.")


if __name__ == "__main__":
    unittest.main()
