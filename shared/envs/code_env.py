"""
Code Environment for RLVR
代码修复/生成环境，使用单元测试作为验证器
"""

import subprocess
import tempfile
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .base import (
    BaseEnv, EnvConfig, Action, ActionType, 
    Observation, Step, Trajectory, VerifierInfo
)


@dataclass
class CodeTask:
    """代码任务定义"""
    task_id: str
    prompt: str
    initial_code: str = ""
    test_code: str = ""
    expected_output: Optional[str] = None
    language: str = "python"
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeEnv(BaseEnv):
    """
    代码环境：支持代码编写、执行、单元测试验证
    """
    
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self.current_task: Optional[CodeTask] = None
        self.current_code: str = ""
        self.execution_history: List[Dict[str, Any]] = []
        self._steps: List[Step] = []
        self._step_counter: int = 0
        self._start_time: float = 0.0
        self._total_tokens: int = 0
        self._total_tool_calls: int = 0
        
    def reset(self, task: Dict[str, Any]) -> Observation:
        """重置环境到新任务"""
        self.clear_cache()
        self.current_task = CodeTask(
            task_id=task.get("task_id", str(uuid.uuid4())),
            prompt=task["prompt"],
            initial_code=task.get("initial_code", ""),
            test_code=task.get("test_code", ""),
            expected_output=task.get("expected_output"),
            language=task.get("language", "python"),
            timeout=task.get("timeout", 30.0),
            metadata=task.get("metadata", {}),
        )
        
        self.current_code = self.current_task.initial_code
        self.execution_history = []
        self._steps = []
        self._step_counter = 0
        self._start_time = time.time()
        self._total_tokens = 0
        self._total_tool_calls = 0
        
        # 构建初始观察
        obs_content = f"Task: {self.current_task.prompt}\n\n"
        if self.current_task.initial_code:
            obs_content += f"Initial Code:\n```{self.current_task.language}\n{self.current_task.initial_code}\n```\n\n"
        if self.current_task.test_code:
            obs_content += f"Test Cases:\n```{self.current_task.language}\n{self.current_task.test_code}\n```"
        
        return Observation(
            content=obs_content,
            obs_type="task_description",
            metadata={"task_id": self.current_task.task_id}
        )
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        self._step_counter += 1
        if action.action_type == ActionType.TOOL_CALL:
            self._total_tool_calls += 1
        
        if action.action_type == ActionType.CODE_WRITE:
            obs, reward, done, info = self._handle_code_write(action)
        elif action.action_type == ActionType.CODE_EXECUTE:
            obs, reward, done, info = self._handle_code_execute(action)
        elif action.action_type == ActionType.TOOL_CALL:
            obs, reward, done, info = self._handle_tool_call(action)
        else:
            obs = Observation(
                content=f"Unknown action type: {action.action_type}",
                obs_type="error"
            )
            reward, done, info = 0.0, False, {"error": "unknown_action"}
        
        # 记录步骤
        step = Step(
            step_id=self._step_counter,
            observation=obs,
            action=action,
            logprob=action.metadata.get("logprob"),
            tokens=action.metadata.get("tokens"),
            token_logprobs=action.metadata.get("token_logprobs")
        )
        self._steps.append(step)
        
        # 检查是否达到最大步数
        if self._step_counter >= self.config.max_steps:
            done = True
            info["terminated_reason"] = "max_steps"
        
        return obs, reward, done, info
    
    def _handle_code_write(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """处理代码编写动作"""
        self.current_code = action.content
        
        obs = Observation(
            content=f"Code updated successfully.\n\nCurrent code:\n```{self.current_task.language}\n{self.current_code}\n```",
            obs_type="code_update"
        )
        
        return obs, 0.0, False, {"action": "code_write"}
    
    def _handle_code_execute(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """处理代码执行动作"""
        code_to_run = action.content if action.content else self.current_code
        
        # 检查缓存
        cache_key = f"exec_{hash(code_to_run)}"
        cached_result = self.get_cached_tool_output(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = self._execute_code(code_to_run)
        self.execution_history.append(result)
        
        if result["success"]:
            obs_content = f"Execution successful.\nOutput:\n{result['stdout']}"
        else:
            obs_content = f"Execution failed.\nError:\n{result['stderr']}"
        
        obs = Observation(
            content=obs_content,
            obs_type="execution_result",
            metadata=result
        )
        
        # 缓存结果
        response = (obs, 0.0, False, {"action": "code_execute", "result": result})
        self.cache_tool_output(cache_key, response)
        
        return response
    
    def _handle_tool_call(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """处理工具调用"""
        tool_name = action.tool_name
        tool_args = action.tool_args or {}
        
        if tool_name == "run_tests":
            return self._run_tests()
        elif tool_name == "get_code":
            obs = Observation(
                content=f"```{self.current_task.language}\n{self.current_code}\n```",
                obs_type="code_content"
            )
            return obs, 0.0, False, {"action": "get_code"}
        else:
            obs = Observation(
                content=f"Unknown tool: {tool_name}",
                obs_type="error"
            )
            return obs, 0.0, False, {"error": "unknown_tool"}
    
    def _run_tests(self) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """运行测试用例"""
        if not self.current_task.test_code:
            obs = Observation(content="No test cases available.", obs_type="info")
            return obs, 0.0, False, {"action": "run_tests", "no_tests": True}
        
        # 组合代码和测试
        full_code = f"{self.current_code}\n\n{self.current_task.test_code}"
        
        # 检查缓存
        cache_key = f"test_{hash(full_code)}"
        cached_result = self.get_cached_tool_output(cache_key)
        if cached_result is not None:
            return cached_result
        
        result = self._execute_code(full_code)
        
        # 解析测试结果
        verifier_info = self._parse_test_result(result)
        
        if verifier_info.success:
            obs_content = f"All tests passed! ({verifier_info.passed_tests}/{verifier_info.total_tests})"
            reward = 1.0
            done = True
        else:
            obs_content = f"Tests failed ({verifier_info.passed_tests}/{verifier_info.total_tests}).\n"
            if result["stderr"]:
                obs_content += f"Error:\n{result['stderr']}"
            reward = 0.0
            done = False
        
        obs = Observation(
            content=obs_content,
            obs_type="test_result",
            metadata={"verifier_info": verifier_info.__dict__}
        )
        
        response = (obs, reward, done, {
            "action": "run_tests", 
            "verifier_info": verifier_info,
            "result": result
        })
        self.cache_tool_output(cache_key, response)
        
        return response
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """执行代码"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.current_task.timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
        finally:
            os.unlink(temp_path)
    
    def _parse_test_result(self, result: Dict[str, Any]) -> VerifierInfo:
        """解析测试结果"""
        # 简单的测试结果解析
        # 实际使用时可以解析pytest/unittest的输出
        if result["success"]:
            return VerifierInfo(
                success=True,
                score=1.0,
                passed_tests=1,
                total_tests=1
            )
        else:
            return VerifierInfo(
                success=False,
                score=0.0,
                error_message=result["stderr"],
                error_type="test_failure",
                passed_tests=0,
                total_tests=1
            )
    
    def verify(self) -> VerifierInfo:
        """验证当前代码是否通过所有测试"""
        if not self.current_task.test_code:
            # 没有测试用例，检查是否有期望输出
            if self.current_task.expected_output:
                result = self._execute_code(self.current_code)
                if result["success"] and result["stdout"].strip() == self.current_task.expected_output.strip():
                    return VerifierInfo(success=True, score=1.0)
                else:
                    return VerifierInfo(
                        success=False, 
                        score=0.0,
                        error_message="Output mismatch",
                        diff_info=f"Expected: {self.current_task.expected_output}\nGot: {result['stdout']}"
                    )
            return VerifierInfo(success=False, score=0.0, error_message="No verification criteria")
        
        # 运行测试
        _, _, _, info = self._run_tests()
        return info.get("verifier_info", VerifierInfo(success=False, score=0.0))
    
    def get_trajectory(self) -> Trajectory:
        """获取当前轨迹"""
        verifier_info = self.verify()
        
        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            task_id=self.current_task.task_id,
            prompt=self.current_task.prompt,
            steps=self._steps.copy(),
            r_final=1.0 if verifier_info.success else 0.0,
            verifier_info=verifier_info,
            total_tokens=self._total_tokens,
            total_tool_calls=self._total_tool_calls,
            wall_time_seconds=time.time() - self._start_time,
            metadata={
                **({"task_type": self.current_task.metadata.get("task_type")} if self.current_task.metadata else {}),
                "language": self.current_task.language,
                "final_code": self.current_code,
                "task": {
                    "task_id": self.current_task.task_id,
                    "prompt": self.current_task.prompt,
                    "initial_code": self.current_task.initial_code,
                    "test_code": self.current_task.test_code,
                    "expected_output": self.current_task.expected_output,
                    "language": self.current_task.language,
                    "timeout": self.current_task.timeout,
                    "metadata": self.current_task.metadata,
                },
            }
        )
