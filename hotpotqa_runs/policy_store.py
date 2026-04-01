"""
PolicyStore — shared across all 3 tasks.

Maintains one PolicyDocument per key (error_class for HotPotQA/Programming,
task_type for ALFWorld). Policy is initialised on first failure of a type
and refined after every subsequent failure.
"""

from typing import List, Dict, Optional


class PolicyDocument:
    """Structured, versioned policy for a specific task type or error class."""

    def __init__(self, key: str):
        self.key             = key
        self.strategy        = ""        # high-level approach
        self.key_steps: List[str] = []   # ordered action steps
        self.avoid:     List[str] = []   # common mistakes to avoid
        self.version         = 0         # number of refinements

    def is_initialised(self) -> bool:
        return bool(self.strategy)

    def to_prompt_str(self) -> str:
        if not self.is_initialised():
            return ""
        steps = '\n'.join(f'  {i+1}. {s}' for i, s in enumerate(self.key_steps))
        avoid = '\n'.join(f'  - {a}' for a in self.avoid)
        return (
            f"=== LEARNED POLICY FOR '{self.key}' (v{self.version}) ===\n"
            f"Strategy: {self.strategy}\n"
            + (f"Key steps:\n{steps}\n" if steps else "")
            + (f"Avoid:\n{avoid}\n" if avoid else "")
            + "=== END POLICY ===\n"
        )

    def update_from_raw(self, raw: str) -> None:
        """Parse LLM response and update policy fields."""
        self.key_steps = []
        self.avoid     = []
        for line in raw.strip().split('\n'):
            line = line.strip()
            if line.upper().startswith('STRATEGY:'):
                s = line.split(':', 1)[-1].strip()
                if s:
                    self.strategy = s
            elif line.upper().startswith('STEP_') or (line and line[0].isdigit() and '.' in line[:3]):
                s = line.split(':', 1)[-1].strip() if ':' in line else line.split('.', 1)[-1].strip()
                if s:
                    self.key_steps.append(s)
            elif line.upper().startswith('AVOID:'):
                s = line.split(':', 1)[-1].strip()
                if s:
                    self.avoid.append(s)
        self.version += 1


class PolicyStore:
    """
    Stores one PolicyDocument per key.
    Keys are error_class strings (HotPotQA, Programming)
    or task_type strings (ALFWorld).
    """

    def __init__(self):
        self.policies: Dict[str, PolicyDocument] = {}

    def get(self, key: str) -> PolicyDocument:
        if key not in self.policies:
            self.policies[key] = PolicyDocument(key)
        return self.policies[key]

    def update(self, key: str, trajectory: str, reflection: str, llm_fn) -> None:
        """
        Initialise or refine the policy for `key` using the current
        failed trajectory and its reflection.
        llm_fn: callable(prompt: str) -> str
        """
        policy = self.get(key)

        if not policy.is_initialised():
            prompt = (
                f"Based on this failed trajectory and reflection, "
                f"write an initial policy for handling '{key}' failures.\n\n"
                f"Trajectory:\n{trajectory[:800]}\n\n"
                f"Reflection:\n{reflection}\n\n"
                "Write the policy in EXACTLY this format:\n"
                "STRATEGY: <one sentence high-level approach>\n"
                "STEP_1: <first action to take>\n"
                "STEP_2: <second action to take>\n"
                "STEP_3: <third action if needed>\n"
                "AVOID: <most common mistake to avoid>\n"
            )
        else:
            prompt = (
                f"Current policy for '{key}':\n{policy.to_prompt_str()}\n\n"
                f"This policy still led to a failure:\n{trajectory[:600]}\n\n"
                f"Reflection on this failure:\n{reflection}\n\n"
                "Refine the policy to fix this failure. Keep what works, fix what doesn't.\n"
                "Use EXACTLY this format:\n"
                "STRATEGY: <updated approach>\n"
                "STEP_1: <first action>\n"
                "STEP_2: <second action>\n"
                "STEP_3: <third action if needed>\n"
                "AVOID: <updated mistake to avoid>\n"
            )

        raw = llm_fn(prompt)
        policy.update_from_raw(raw)
        print(f"  Policy '{key}' updated to v{policy.version}: {policy.strategy[:80]}")