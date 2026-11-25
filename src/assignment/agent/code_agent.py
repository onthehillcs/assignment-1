"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        self.system_prompt = (
            "You are a coding agent tasked with fixing a software issue. "
            "Inspect the repository, make focused changes, and run relevant tests. "
            "Use the execute tool to inspect and modify files and run commands "
            "in the sandbox. Inspect tool results before deciding the next action. "
            "Use send_message only for final submission; it ends the task.\n\n"
            f"The sandbox's default working directory is {self.env.cwd}.\n\n"
            f"<system_information>\n{system_information}\n</system_information>"
        )
        self.task_prompt = self.task
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions.\n\n"
                f"<skills>\n{catalog}\n</skills>\n"
            )

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute coding tools and load available skill instructions."""

        messages = []
        for call in tool_calls:
            try:
                if self.finished:
                    raise ValueError("Task already submitted; no further tools may run")
                function = call["function"]
                name = function["name"]
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be a JSON object")

                if name == "execute":
                    content = format_tool_output(self.env.execute(**arguments))
                elif name == "invoke_skill":
                    skill_name = arguments["name"]
                    if not isinstance(skill_name, str):
                        raise ValueError("name must be a string")
                    if skill_name not in self.skills:
                        raise ValueError(f"Unknown skill {skill_name!r}")
                    content = self.skills[skill_name]["content"]
                elif name == "send_message":
                    summary = arguments["summary"]
                    if not isinstance(summary, str):
                        raise ValueError("summary must be a string")
                    print(summary)
                    content = "Submission received. Task complete."
                    self.finished = True
                else:
                    content = f"Error: Unknown tool {name!r}"
            except (KeyError, TypeError, ValueError) as exc:
                content = f"Error: {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                }
            )
        return messages
