"""Qwen bridge used only on the target side of the pipeline.

Scope:
- Parse target questions for LVQA/ME-VQA.
- Render ME-VQA analysis text from an already-built profile.
- Never touch training annotation loading, reference reconstruction, priors, or profile synthesis logic.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Optional


class QwenBridge:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens: int = 128,
    ):
        """Initialize and validate the Qwen interface at startup.

        Args:
            model_name: Hugging Face model id or local model path.
            max_new_tokens: Default generation cap for bridge prompts.
        Returns:
            None.
        When used:
            Called once by root entry scripts before target-side parsing/decoding starts.
        """
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self._pipe = None
        self._load_error = None
        self._ensure_pipe()

    def _ensure_pipe(self):
        """Load and cache the text-generation pipeline for Qwen.

        Args:
            None.
        Returns:
            Initialized transformers pipeline object.
        When used:
            During bridge initialization and before each Qwen-backed call.
        """
        if self._pipe is not None:
            return self._pipe
        if self._load_error is not None:
            raise RuntimeError(
                f"Qwen initialization previously failed for model '{self.model_name}'."
            ) from self._load_error
        try:
            from transformers import pipeline  # type: ignore

            self._pipe = pipeline(
                task="text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            return self._pipe
        except Exception as exc:
            self._load_error = exc
            raise RuntimeError(
                f"Failed to initialize mandatory Qwen model '{self.model_name}'. "
                "Check model path/id and transformers runtime dependencies."
            ) from exc

    @staticmethod
    def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
        """Extract one JSON object from raw model text.

        Args:
            text: Raw generated text that should contain JSON.
        Returns:
            Parsed dictionary, or None when extraction/parsing fails.
        When used:
            Immediately after Qwen generation to decode structured output.
        """
        if not text:
            return None
        s = text.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def _run_json_prompt(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Run a constrained Qwen prompt that must return JSON.

        Args:
            system_prompt: Instruction describing output schema and constraints.
            user_prompt: Task-specific input payload.
            max_new_tokens: Optional override for generation length.
        Returns:
            Parsed JSON dictionary, or None when generation/parsing fails.
        When used:
            Shared low-level helper for LVQA parse, ME-VQA parse, and analysis rendering.
        """
        pipe = self._ensure_pipe()

        prompt = f"{system_prompt}\n\n{user_prompt}\n\nReturn JSON only."
        try:
            outputs = pipe(
                prompt,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
            )
            if not outputs:
                return None
            text = outputs[0].get("generated_text", "")
            return self._extract_json_dict(str(text))
        except Exception:
            return None

    @staticmethod
    def _sanitize_lvqa_parse(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate LVQA parse schema produced by Qwen.

        Args:
            obj: Candidate parsed JSON object.
        Returns:
            Normalized parse dictionary, or None when invalid.
        When used:
            After Qwen parses target LVQA questions.
        """
        kind = str(obj.get("kind", "")).strip().lower()
        allowed = {"total", "micro", "macro", "nth_type", "au", "unknown"}
        if kind not in allowed:
            return None
        if kind == "nth_type":
            idx = obj.get("index")
            try:
                ii = int(idx)
            except Exception:
                return None
            if ii <= 0:
                return None
            return {"kind": "nth_type", "index": ii}
        return {"kind": kind}

    @staticmethod
    def _sanitize_mevqa_parse(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate ME-VQA parse schema produced by Qwen.

        Args:
            obj: Candidate parsed JSON object.
        Returns:
            Normalized parse dictionary, or None when invalid.
        When used:
            After Qwen parses target ME-VQA questions.
        """
        kind = str(obj.get("kind", "")).strip().lower()
        allowed = {"coarse", "fine", "single_au", "multi_au", "bool_au", "analysis", "unknown"}
        if kind not in allowed:
            return None
        if kind == "bool_au":
            au = str(obj.get("au", "")).strip().lower()
            if not au:
                return None
            return {"kind": "bool_au", "au": au}
        return {"kind": kind}

    def parse_lvqa_question(self, question: str, fallback_parser: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        """Parse one target LVQA question into the internal requirement schema.

        Args:
            question: Raw target question text.
            fallback_parser: Rule-based parser used when Qwen output is invalid.
        Returns:
            Dictionary with fields like {"kind": "..."} and optional "index".
        When used:
            Stage 3 requirement extraction for ME-LVQA target rows.
        """
        system = (
            "You map a ME-LVQA question into JSON schema.\n"
            'Valid kinds: total, micro, macro, nth_type, au, unknown.\n'
            'For nth_type include integer field "index".'
        )
        user = f'Question: "{question}"\nOutput example: {{"kind":"nth_type","index":3}}'
        obj = self._run_json_prompt(system, user, max_new_tokens=80)
        if not obj:
            return fallback_parser(question)
        parsed = self._sanitize_lvqa_parse(obj)
        if not parsed:
            return fallback_parser(question)
        return parsed

    def parse_mevqa_question(self, question: str, fallback_parser: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
        """Parse one target ME-VQA question into the internal requirement schema.

        Args:
            question: Raw target question text.
            fallback_parser: Rule-based parser used when Qwen output is invalid.
        Returns:
            Dictionary with parsed ME-VQA question intent and optional AU slot.
        When used:
            Stage 3 requirement extraction for ME-VQA target rows.
        """
        system = (
            "You map a ME-VQA question into JSON schema.\n"
            'Valid kinds: coarse, fine, single_au, multi_au, bool_au, analysis, unknown.\n'
            'For bool_au include string field "au".'
        )
        user = f'Question: "{question}"\nOutput example: {{"kind":"bool_au","au":"lip corner puller"}}'
        obj = self._run_json_prompt(system, user, max_new_tokens=80)
        if not obj:
            return fallback_parser(question)
        parsed = self._sanitize_mevqa_parse(obj)
        if not parsed:
            return fallback_parser(question)
        return parsed

    def render_mevqa_analysis(self, question: str, profile: Dict[str, Any], fallback_renderer: Callable[[str, Dict[str, Any]], str]) -> str:
        """Render target ME-VQA analysis text from an existing pseudo-profile.

        Args:
            question: Original analysis-style target question.
            profile: Fixed pseudo-profile containing coarse/fine/AU outputs.
            fallback_renderer: Deterministic renderer used on invalid Qwen output.
        Returns:
            One analysis answer string aligned with the given profile.
        When used:
            Stage 5 answer decoding for ME-VQA analysis questions only.
        """

        coarse = str(profile.get("coarse", "negative"))
        fine = str(profile.get("fine", "disgust"))
        aus = profile.get("aus", []) or []
        if not isinstance(aus, list):
            aus = []
        au_text = ", ".join(str(a) for a in aus)

        system = (
            "Write one concise analysis sentence using only provided profile facts.\n"
            "Do not invent labels or AUs.\n"
            "If AU list is empty, mention subtle movement.\n"
            "Return JSON with key text."
        )
        user = (
            f'Question: "{question}"\n'
            f"Profile coarse: {coarse}\n"
            f"Profile fine: {fine}\n"
            f"Profile aus: {au_text if au_text else '(none)'}\n"
            'Output JSON: {"text":"..."}'
        )
        obj = self._run_json_prompt(system, user, max_new_tokens=120)
        if not obj:
            return fallback_renderer(question, profile)
        text = obj.get("text")
        if not isinstance(text, str):
            return fallback_renderer(question, profile)
        text = text.strip()
        if not text:
            return fallback_renderer(question, profile)

        # Basic safety: ensure coarse/fine from profile appear in output to avoid drift.
        low = text.lower()
        if coarse.lower() not in low or fine.lower() not in low:
            return fallback_renderer(question, profile)
        return text
