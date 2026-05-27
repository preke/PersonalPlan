from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Learner(BaseModel):
    self_description: str
    skills: List[str] = Field(default_factory=list)


class InputPayload(BaseModel):
    query: str
    learner: Learner
    accepted_answer: Optional[str] = None


class AgentSpec(BaseModel):
    agent_role: str
    goal: str
    description: str
    tools: List[str] = Field(default_factory=list)


class StepSpec(BaseModel):
    id: str
    agent: str
    objective: str
    instruction: str
    tool: Optional[str] = None
    requires_human_input: bool = False
    expected_output: str
    depends_on: List[str] = Field(default_factory=list)


class SubtaskSpec(BaseModel):
    id: str
    name: str
    subtask_objective: str
    steps: List[StepSpec]


class LoopSpec(BaseModel):
    steps: List[str] = Field(default_factory=list)
    step: Optional[str] = None
    condition: str
    max_iterations: int = 1

    def model_post_init(self, __context: Any) -> None:
        """If 'step' (singular) is provided but 'steps' is empty, convert it."""
        if self.step and not self.steps:
            self.steps = [self.step]
        if not self.steps:
            raise ValueError("LoopSpec requires either 'steps' or 'step' to be provided.")


class LoopBlock(BaseModel):
    loop: LoopSpec


ExecutionItem = Union[str, LoopBlock]


class OutputPayload(BaseModel):
    agents: List[AgentSpec]
    subtasks: List[SubtaskSpec]
    execution_order: List[ExecutionItem]


class PlanPayload(BaseModel):
    input: InputPayload
    output: OutputPayload


class StepRunResult(BaseModel):
    step_id: str
    status: Literal["PASS", "FAIL", "SKIPPED", "ERROR"]
    content: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    mode: Literal["smoke", "live"] = "smoke"
    model: Optional[str] = None
    student_model: Optional[str] = None
    run_id: str


class CompileReport(BaseModel):
    step_count: int
    agent_count: int
    loop_count: int
    dependency_errors: List[str] = Field(default_factory=list)
    tool_binding_errors: List[str] = Field(default_factory=list)


class ConformanceReport(BaseModel):
    passed: bool
    immutable_diffs: List[str] = Field(default_factory=list)


class ExecutionReport(BaseModel):
    run_id: str
    mode: Literal["smoke", "live"]
    succeeded: bool
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    loop_events: List[Dict[str, Any]] = Field(default_factory=list)
