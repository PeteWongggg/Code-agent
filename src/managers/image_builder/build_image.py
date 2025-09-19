"""
SWE-bench 数据集加载器
支持加载数据集并克隆对应的仓库到工作空间
"""

import os
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional, Any
from datasets import load_dataset

import docker
from swebench.harness.prepare_images import filter_dataset_to_build
from swebench.harness.docker_build import build_instance_images
from swebench.harness.utils import load_swebench_dataset
from swebench.harness.test_spec.test_spec import make_test_spec


class SWEBenchLoader:
    """
    SWE-bench 数据集加载器
    
    功能：
    - 加载 SWE-bench 数据集
    - 遍历数据集中的每条数据
    - 根据 instance_id 创建子文件夹
    - 克隆对应的 Git 仓库
    """
    
    def __init__(
        self,
        dataset_name: str = 'princeton-nlp/SWE-bench_Lite',
        split_name: str = 'dev',
        workspace_path: str = 'workspace',
        logger: Optional[Any] = None
    ):
        """
        初始化数据加载器
        
        Args:
            dataset_name: 数据集名称
            split_name: 数据集分割名称
            workspace_path: 工作空间路径
            logger: 日志记录器
        """
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.workspace_path = Path(workspace_path)
        self.logger = logger
        self.dataset = None
        
        # 创建工作空间目录
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"初始化 SWE-bench 数据加载器 - 数据集: {dataset_name}, 分割: {split_name}, 工作空间: {workspace_path}")
    
    def _convert_repo_to_url(self, repo_name: str) -> str:
        """
        将仓库名称转换为完整的 GitHub URL
        
        Args:
            repo_name: 仓库名称，格式如 'owner/repo'
            
        Returns:
            str: 完整的 GitHub URL
        """
        if repo_name.startswith('http'):
            # 如果已经是完整URL，直接返回
            return repo_name
        else:
            # 转换为 GitHub URL
            return f"https://github.com/{repo_name}.git"
    
    def load_dataset(self):
        """
        加载整体数据集
        
        Returns:
            Dataset: 加载的数据集
        """
        try:
            if self.logger:
                self.logger.info(f"开始加载数据集: {self.dataset_name}, 分割: {self.split_name}")
            
            self.dataset = load_dataset(self.dataset_name, split=self.split_name)
            
            if self.logger:
                self.logger.info(f"成功加载数据集，包含 {len(self.dataset)} 条数据")
            
            return self.dataset
            
        except Exception as e:
            error_msg = f"加载数据集失败: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def clone_repository(self, repo_url: str, target_path: Path, commit_hash: Optional[str] = None, 
                        max_retries: int = 3, retry_delay: int = 5, force_reclone: bool = True) -> bool:
        """
        克隆 Git 仓库到指定路径（支持重试机制）
        
        Args:
            repo_url: 仓库 URL
            target_path: 目标路径
            commit_hash: 特定的提交哈希（可选）
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
            force_reclone: 是否强制重新克隆（删除已存在的仓库）
            
        Returns:
            bool: 克隆是否成功
        """
        for attempt in range(max_retries + 1):
            try:
                # 如果强制重新克隆且目标路径已存在，先删除
                if force_reclone and target_path.exists():
                    if self.logger:
                        self.logger.debug(f"删除已存在的仓库: {target_path}")
                    shutil.rmtree(target_path)
                
                # 如果目标路径已存在且不强制重新克隆，检查是否为有效的git仓库
                elif target_path.exists() and (target_path / '.git').exists():
                    if self.logger:
                        self.logger.debug(f"仓库已存在，跳过克隆: {target_path}")
                    
                    # 如果指定了commit hash，尝试切换
                    if commit_hash:
                        return self._checkout_commit(target_path, commit_hash)
                    return True
                
                # 克隆仓库
                if self.logger:
                    self.logger.debug(f"开始克隆仓库 (尝试 {attempt + 1}/{max_retries + 1}): {repo_url} -> {target_path}")
                
                cmd = ['git', 'clone', repo_url, str(target_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    error_msg = f"Git clone 失败 (尝试 {attempt + 1}): {result.stderr.strip()}"
                    if self.logger:
                        self.logger.warning(error_msg)
                    
                    # 如果不是最后一次尝试，等待后重试
                    if attempt < max_retries:
                        if self.logger:
                            self.logger.info(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        if self.logger:
                            self.logger.error(f"所有重试都失败，无法克隆仓库: {repo_url}")
                        return False
                
                # 如果指定了 commit hash，切换到该提交
                if commit_hash:
                    if not self._checkout_commit(target_path, commit_hash):
                        # checkout失败但不影响克隆成功
                        if self.logger:
                            self.logger.warning(f"克隆成功但切换提交失败: {commit_hash}")
                
                if self.logger:
                    self.logger.info(f"成功克隆仓库: {repo_url}")
                
                return True
                
            except subprocess.TimeoutExpired:
                error_msg = f"Git 操作超时 (尝试 {attempt + 1}): {repo_url}"
                if self.logger:
                    self.logger.warning(error_msg)
                
                # 清理可能的部分克隆
                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries:
                    if self.logger:
                        self.logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if self.logger:
                        self.logger.error(f"所有重试都因超时失败: {repo_url}")
                    return False
                    
            except Exception as e:
                error_msg = f"克隆仓库异常 (尝试 {attempt + 1}): {str(e)}"
                if self.logger:
                    self.logger.warning(error_msg)
                
                # 清理可能的部分克隆
                if target_path.exists():
                    shutil.rmtree(target_path, ignore_errors=True)
                
                # 如果不是最后一次尝试，等待后重试
                if attempt < max_retries:
                    if self.logger:
                        self.logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    if self.logger:
                        self.logger.error(f"所有重试都因异常失败: {repo_url}")
                    return False
        
        return False
    
    def _checkout_commit(self, repo_path: Path, commit_hash: str) -> bool:
        """
        切换到指定的提交
        
        Args:
            repo_path: 仓库路径
            commit_hash: 提交哈希
            
        Returns:
            bool: 切换是否成功
        """
        try:
            if self.logger:
                self.logger.debug(f"切换到提交: {commit_hash}")
            
            cmd = ['git', 'checkout', commit_hash]
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                if self.logger:
                    self.logger.warning(f"切换提交失败: {result.stderr.strip()}")
                return False
            
            if self.logger:
                self.logger.debug(f"成功切换到提交: {commit_hash}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"切换提交异常: {str(e)}")
            return False
    
    def process_dataset_item(self, item: dict) -> bool:
        """
        处理数据集中的单个条目
        
        Args:
            item: 数据集条目
            
        Returns:
            bool: 处理是否成功
        """
        try:
            instance_id = item.get('instance_id')
            repo_name = item.get('repo')  # 格式如 'sqlfluff/sqlfluff'
            base_commit = item.get('base_commit')
            
            if not instance_id:
                if self.logger:
                    self.logger.error("数据条目缺少 instance_id")
                return False
            
            if not repo_name:
                if self.logger:
                    self.logger.error(f"数据条目 {instance_id} 缺少 repo 信息")
                return False
            
            # 将仓库名称转换为完整的 GitHub URL
            repo_url = self._convert_repo_to_url(repo_name)
            
            if self.logger:
                self.logger.debug(f"处理实例 {instance_id}: {repo_name} -> {repo_url}")
            
            # 创建实例文件夹
            instance_dir = self.workspace_path / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            
            # 克隆仓库
            repo_dir = instance_dir / "repository"
            success = self.clone_repository(repo_url, repo_dir, base_commit)
            
            if success:
                # 保存实例信息到文件
                info_file = instance_dir / "instance_info.txt"
                with open(info_file, 'w', encoding='utf-8') as f:
                    f.write(f"Instance ID: {instance_id}\n")
                    f.write(f"Repository Name: {repo_name}\n")
                    f.write(f"Repository URL: {repo_url}\n")
                    f.write(f"Base Commit: {base_commit}\n")
                    f.write(f"Problem Statement: {item.get('problem_statement', '')}\n")
                    f.write(f"Created At: {item.get('created_at', '')}\n")
                
                if self.logger:
                    self.logger.info(f"成功处理实例: {instance_id}")
                return True
            else:
                if self.logger:
                    self.logger.error(f"处理实例失败: {instance_id}")
                return False
                
        except Exception as e:
            error_msg = f"处理数据条目异常: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            return False
    
    def load_and_process_all(self, max_items: Optional[int] = None) -> dict:
        """
        加载数据集并处理所有条目
        
        Args:
            max_items: 最大处理条目数（可选，用于测试）
            
        Returns:
            dict: 处理结果统计
        """
        # 加载数据集
        if self.dataset is None:
            self.load_dataset()
        
        total_items = len(self.dataset)
        if max_items:
            total_items = min(total_items, max_items)
        
        successful_items = 0
        failed_items = 0
        
        if self.logger:
            self.logger.info(f"开始处理 {total_items} 个数据条目")
        
        # 遍历处理每个条目
        for i, item in enumerate(self.dataset):
            if max_items and i >= max_items:
                break
            
            if self.logger:
                self.logger.debug(f"处理第 {i+1}/{total_items} 个条目: {item.get('instance_id', 'unknown')}")
            
            success = self.process_dataset_item(item)
            
            if success:
                successful_items += 1
            else:
                failed_items += 1
        
        # 统计结果
        result = {
            'total_processed': total_items,
            'successful': successful_items,
            'failed': failed_items,
            'success_rate': (successful_items / total_items * 100) if total_items > 0 else 0
        }
        
        if self.logger:
            self.logger.info(f"处理完成 - 总数: {result['total_processed']}, "
                           f"成功: {result['successful']}, 失败: {result['failed']}, "
                           f"成功率: {result['success_rate']:.1f}%")
        
        return result
    
    def get_instance_path(self, instance_id: str) -> Path:
        """
        获取指定实例的路径
        
        Args:
            instance_id: 实例 ID
            
        Returns:
            Path: 实例路径
        """
        return self.workspace_path / instance_id
    
    def get_repository_path(self, instance_id: str) -> Path:
        """
        获取指定实例的仓库路径
        
        Args:
            instance_id: 实例 ID
            
        Returns:
            Path: 仓库路径
        """
        return self.workspace_path / instance_id / "repository"
    
    def list_instances(self) -> list:
        """
        列出工作空间中的所有实例
        
        Returns:
            list: 实例 ID 列表
        """
        instances = []
        if self.workspace_path.exists():
            for item in self.workspace_path.iterdir():
                if item.is_dir():
                    instances.append(item.name)
        return sorted(instances)
    
    def get_stats(self) -> dict:
        """
        获取工作空间统计信息
        
        Returns:
            dict: 统计信息
        """
        instances = self.list_instances()
        cloned_repos = 0
        
        for instance_id in instances:
            repo_path = self.get_repository_path(instance_id)
            if repo_path.exists() and (repo_path / '.git').exists():
                cloned_repos += 1
        
        return {
            'total_instances': len(instances),
            'cloned_repositories': cloned_repos,
            'workspace_path': str(self.workspace_path.absolute())
        }

class SWEBenchImageBuilder:
    def __init__(
        self,
        dataset_name: str = "SWE-bench/SWE-bench_Lite",
        split: str = "test",
        instance_ids: list = None,
        max_workers: int = 4,
        force_rebuild: bool = False,
        namespace: str = None,
        tag: str = "latest",
        env_image_tag: str = "latest",
    ):
        """
        Initialize the image builder and build all required images.
        
        Args:
            dataset_name: Name of the dataset to use
            split: Split to use (dev/test)
            instance_ids: List of instance IDs to build (None for all)
            max_workers: Number of workers for parallel processing
            force_rebuild: Whether to force rebuild all images
            namespace: Namespace for images
            tag: Tag for images (default: "latest")
            env_image_tag: Environment image tag (default: "latest")
        """
        self.client = docker.from_env()
        self.dataset_name = dataset_name
        self.split = split
        self.namespace = namespace
        self.tag = tag
        self.env_image_tag = env_image_tag
        self.instance_to_image = {}
        self.full_dataset = load_swebench_dataset(dataset_name, split)
        self._populate_existing_images_mapping(instance_ids)
        
        self.dataset_to_build = filter_dataset_to_build(
            self.full_dataset, instance_ids, self.client, force_rebuild, namespace, tag, self.env_image_tag
        )
        
        if self.instance_to_image:
            print(f"Filtering out {len(self.instance_to_image)} existing instances from dataset_to_build...")
            # Create a set of existing instance IDs for faster lookup
            existing_instance_ids = set(self.instance_to_image.keys())
            # Filter out instances that already have images
            self.dataset_to_build = [inst for inst in self.dataset_to_build if inst['instance_id'] not in existing_instance_ids]
            print(f"Remaining instances to process: {len(self.dataset_to_build)}")
        
        if len(self.dataset_to_build) == 0:
            print("All images exist. Nothing left to build.")
            self.successful = []
            self.failed = []
            print(f"Populating mapping for instance_ids: {instance_ids}")
        else:
            print(f"Building images for {len(self.dataset_to_build)} instances...")
            self.successful, self.failed = build_instance_images(
                client=self.client,
                dataset=self.dataset_to_build,
                force_rebuild=force_rebuild,
                max_workers=max_workers,
                namespace=namespace,
                tag=tag,
                env_image_tag=self.env_image_tag,
            )
            print(f"Successfully built {len(self.successful)} images")
            print(f"Failed to build {len(self.failed)} images")
            
            if self.successful:
                print("Adding ripgrep to built images...")
                self._add_ripgrep_to_images(self.successful)
    
    def get_image_name(self, instance_id: str) -> str:
        """
        Get the Docker image name for a given instance_id.
        
        Args:
            instance_id: The instance ID to look up
            
        Returns:
            The Docker image name for the instance
            
        Raises:
            KeyError: If instance_id is not found in the dataset
        """
        if instance_id not in self.instance_to_image:
            raise KeyError(f"Instance ID '{instance_id}' not found in dataset")
        
        return self.instance_to_image[instance_id]
    
    def get_build_status(self, instance_id: str) -> str:
        """
        Get the build status for a given instance_id.
        
        Args:
            instance_id: The instance ID to check
            
        Returns:
            'successful', 'failed', or 'not_built'
        """
        if instance_id not in self.instance_to_image:
            return 'not_found'
        
        # Check if this instance was in the build list
        instance_in_build_list = any(
            inst['instance_id'] == instance_id for inst in self.dataset_to_build
        )
        
        if not instance_in_build_list:
            return 'already_exists'
        
        # Check if it was successfully built
        for successful_instance in self.successful:
            if successful_instance['instance_id'] == instance_id:
                return 'successful'
        
        # Check if it failed to build
        for failed_instance in self.failed:
            if failed_instance['instance_id'] == instance_id:
                return 'failed'
        
        return 'unknown'
    
    def _populate_existing_images_mapping(self, instance_ids):
        """Populate instance_to_image mapping for existing images that don't need rebuilding."""
        if not instance_ids:
            return
            
        for instance_id in instance_ids:
            try:
                # Find the instance in the full dataset
                instance = None
                for inst in self.full_dataset:
                    if inst['instance_id'] == instance_id:
                        instance = inst
                        break
                
                if instance:
                    test_spec = make_test_spec(instance)
                    base_image_name = test_spec.instance_image_key
                    
                    # Modify the image name to look for the ripgrep version
                    if ':' in base_image_name:
                        base_name, tag = base_image_name.rsplit(':', 1)
                        ripgrep_image_name = f"{base_name}-with-ripgrep"
                    else:
                        ripgrep_image_name = f"{base_image_name}-with-ripgrep"
                    
                    try:
                        self.client.images.get(ripgrep_image_name)
                        self.instance_to_image[instance_id] = ripgrep_image_name
                        print(f"Found existing ripgrep image for {instance_id}: {ripgrep_image_name}")
                    except docker.errors.ImageNotFound:
                        print(f"Ripgrep image not found with name {ripgrep_image_name}, will use original image")
                        
            except Exception as e:
                print(f"Error checking existing image for {instance_id}: {e}")

    def _add_ripgrep_to_images(self, successful_images):
        """Add ripgrep to all successfully built images by creating new layers"""
        for successful_instance in successful_images:
            try:
                # Extract image name from the successful instance
                # The successful_instance is a tuple containing (TestSpec, DockerClient, Logger, bool)
                if isinstance(successful_instance, tuple) and len(successful_instance) > 0:
                    test_spec = successful_instance[0]  # First element is the TestSpec
                    old_image_name = test_spec.instance_image_key
                    instance_id = test_spec.instance_id
                else:
                    raise Exception(f"Unexpected successful_instance format: {type(successful_instance)}")
                
                # Check if this instance already has a ripgrep image in the mapping
                if instance_id in self.instance_to_image:
                    existing_image = self.instance_to_image[instance_id]
                    if '-with-ripgrep' in existing_image:
                        print(f"✓ Skipping ripgrep addition for {instance_id}: already has ripgrep image {existing_image}")
                        continue
                
                print(f"Adding ripgrep to {old_image_name}...")
                new_image_name = self._add_ripgrep_to_single_image(old_image_name)
                
                # Update the instance_to_image mapping to point to the new image
                self.instance_to_image[instance_id] = new_image_name
                print(f"✓ Updated mapping for {instance_id}: {old_image_name} -> {new_image_name}")
                
                # Try to delete the old image
                try:
                    self.client.images.remove(old_image_name, force=True)
                    print(f"✓ Deleted old image {old_image_name}")
                except Exception as e:
                    print(f"Warning: Could not delete old image {old_image_name}: {e}")
                
            except Exception as e:
                print(f"✗ Failed to add ripgrep to {successful_instance}: {e}")
    
    def _add_ripgrep_to_single_image(self, old_image_name: str) -> str:
        """Add ripgrep to a single image by creating a new layer. Returns the new image name."""
        try:
            # Create a temporary container from the image
            container = self.client.containers.run(
                old_image_name,
                command="sleep infinity",
                detach=True,
                working_dir="/workspace"
            )
            
            # Install ripgrep in the container - try multiple package managers
            package_managers = [
                # Try apt-get first (Ubuntu/Debian)
                ("apt-get", ["apt-get", "update"], ["apt-get", "install", "-y", "ripgrep"]),
                # Try yum (CentOS/RHEL)
                ("yum", ["yum", "install", "-y", "ripgrep"]),
                # Try dnf (Fedora)
                ("dnf", ["dnf", "install", "-y", "ripgrep"]),
                # Try apk (Alpine)
                ("apk", ["apk", "add", "--no-cache", "ripgrep"]),
                # Try zypper (openSUSE)
                ("zypper", ["zypper", "install", "-y", "ripgrep"])
            ]
            
            exec_result = None
            successful_package_manager = None
            for i, (pm_name, *commands) in enumerate(package_managers):
                print(f"Trying package manager {i+1}/{len(package_managers)}: {pm_name}")
                
                # Execute each command for this package manager
                all_successful = True
                for cmd in commands:
                    exec_result = container.exec_run(cmd, stdout=True, stderr=True)
                    if exec_result.exit_code != 0:
                        print(f"✗ Failed with {pm_name} command {' '.join(cmd)}: {exec_result.output.decode()}")
                        all_successful = False
                        break
                
                if all_successful:
                    print(f"✓ Successfully installed ripgrep using: {pm_name}")
                    successful_package_manager = pm_name
                    break
            
            if not exec_result or exec_result.exit_code != 0:
                raise Exception(f"Failed to install ripgrep with any package manager: {exec_result.output.decode() if exec_result else 'No command executed'}")
            
            # Clean up package manager cache if apt-get was used
            if successful_package_manager == "apt-get":
                print("Cleaning up apt cache...")
                cleanup_result = container.exec_run(["rm", "-rf", "/var/lib/apt/lists/*"], stdout=True, stderr=True)
                if cleanup_result.exit_code != 0:
                    print(f"Warning: Failed to clean up apt cache: {cleanup_result.output.decode()}")
                else:
                    print("✓ Apt cache cleaned up successfully")
            
            # Verify ripgrep installation
            verify_result = container.exec_run("rg --version", stdout=True, stderr=True)
            if verify_result.exit_code != 0:
                raise Exception(f"Ripgrep verification failed: {verify_result.output.decode()}")
            
            # Note: Git repository isolation is already handled by the SWE-bench harness
            # during the initial image build process via the install_repo_script.
            # The harness ensures agents cannot access commits beyond the test commit.
            
            # Create a new image with ripgrep - use a new name
            # Handle image name and tag properly
            if ':' in old_image_name:
                base_name, tag = old_image_name.rsplit(':', 1)
                new_image_name = f"{base_name}-with-ripgrep"
            else:
                new_image_name = f"{old_image_name}-with-ripgrep"
            
            container.commit(repository=new_image_name, tag="latest")
            print(f"✓ Created new image with ripgrep: {new_image_name}")
            
            # Clean up the container
            container.remove(force=True)
            
            # Verify the new image exists
            try:
                new_image = self.client.images.get(f"{new_image_name}:latest")
                print(f"✓ New image verified: {new_image_name} (ID: {new_image.short_id})")
                return f"{new_image_name}:latest"
            except Exception as e:
                raise Exception(f"Failed to verify new image {new_image_name}: {e}")
                
        except Exception as e:
            print(f"Error adding ripgrep to {old_image_name}: {e}")
            raise e
    

if __name__ == '__main__':
    # 测试代码
    swe_loader = SWEBenchLoader(logger=None)
    
    # 加载数据集
    dataset = swe_loader.load_dataset()
    print(f"加载了 {len(dataset)} 条数据")
    
    # 处理前 3 个条目（测试用）
    result = swe_loader.load_and_process_all(max_items=3)
    print(f"处理结果: {result}")
    
    # 显示统计信息
    stats = swe_loader.get_stats()
    print(f"统计信息: {stats}")
