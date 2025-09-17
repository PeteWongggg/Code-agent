import uuid
import docker
import pexpect
import re
from docker.errors import DockerException, ImageNotFound, NotFound

class Executor:
    def __init__(self, image: str):
        self.image = image
        self.container = None
        self.sessions = {}
        self.client = docker.from_env()
        
        try:
            self.client.images.get(self.image)
        except ImageNotFound:
            raise DockerException(f"Image '{self.image}' not found. Please build the image first.")
        
        try:
            self.container = self.client.containers.run(
                self.image,
                command="sleep infinity",
                detach=True,
                working_dir="/workspace"
            )
        except DockerException as e:
            raise DockerException(f"Failed to create container with image '{self.image}': {e}")
        
        session_id = self.init_session()
        if session_id is None:
            raise DockerException("Failed to initialize default session")
        if session_id in self.sessions:
            self.sessions['0'] = self.sessions.pop(session_id)

    def init_session(self) -> str:
        session_id = str(uuid.uuid4())
        command = f"docker exec -it {self.container.id} /bin/bash"
        try:
            shell = pexpect.spawn(command, encoding="utf-8", timeout=120)
            shell.expect([r"\$", r"#"], timeout=120)
            self.sessions[session_id] = shell
            return session_id
        except pexpect.exceptions.TIMEOUT:
            print(f"❌ Timeout waiting for shell prompt.")
            return None

    def execute(self, session_id: str, command: str, timeout: int = 300) -> tuple[int, str]:
        shell = self.sessions.get(session_id)
        if not shell or not shell.isalive():
            print(f"❌ Session {session_id} is not active or does not exist.")
            return -1, "Session not found or is dead."

        marker = f"---CMD_DONE---"
        full_command = command.strip()
        marker_command = f"echo {marker}$?"
        shell.sendline(full_command)
        shell.sendline(marker_command)
        try:
            shell.expect(marker + r"(\d+)", timeout=timeout)
        except pexpect.exceptions.TIMEOUT:
            return -1, f"Error: Command '{command}' timed out after {timeout} seconds. Partial output:\n{shell.before}"

        exit_code = int(shell.match.group(1))
        all_lines = shell.before.splitlines()
        clean_lines = []
        for line in all_lines:
            if line != full_command and marker_command not in line:
                clean_lines.append(line)
        clean_output = "\n".join(clean_lines)
        shell.expect([r"\$", r"#"])
        return exit_code, clean_output.strip()

    def check_session(self) -> bool:
        """
        Check whether the current '0' session is alive and restart it if not.
        """
        if '0' in self.sessions:
            session = self.sessions['0']
            if session and session.isalive():
                print(f"✅ Session {'0'} is alive.")
                return True
            else:
                print(f"❌ Session {'0'} is dead. Cleaning up...")
                self.sessions.pop('0')
        else:
            print(f"❌ Session {'0'} does not exist.")
        
        new_session_id = self.init_session()
        if new_session_id is None:
            print(f"❌ Failed to restart session {'0'}.")
            return False
        if new_session_id != '0':
            self.sessions['0'] = self.sessions.pop(new_session_id)

        return True

    def close_session(self, session_id: str):
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            if session and session.isalive():
                session.close(force=True)
        else:
            print(f"Warning: Session {session_id} not found.")

    def shutdown(self):
        for session_id in list(self.sessions.keys()):
            self.close_session(session_id)
        
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
            except DockerException as e:
                print(f"❌ Could not clean up container: {e}")
        self.container = None