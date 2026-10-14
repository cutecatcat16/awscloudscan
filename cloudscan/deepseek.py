"""Minimal DeepSeek chat-completions client (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from . import config
from .util import (debug, error, info, read_json, thinking_chunk, thinking_end,
                   thinking_header, warn)


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API cannot be reached or returns an error."""


class DeepSeekTruncation(DeepSeekError):
    """The reply hit *max_tokens* (finish_reason=length) and is incomplete.

    Distinct from the transport failures :class:`DeepSeekError` also covers, so a
    caller can tell "the key or the network is broken" apart from "this stage
    needs a bigger budget". Both are fatal to a stage, but only one is fixed by
    raising the budget — and blaming the wrong one sends the operator hunting for
    a connectivity problem that does not exist.
    """


@dataclass
class ChatResult:
    """One completion, plus the thinking that produced it.

    ``reasoning`` is the model's chain of thought: empty when thinking is
    disabled, and billed as ``usage["completion_tokens_details"]["reasoning_tokens"]``.
    ``finish_reason`` is ``"length"`` when the reply hit *max_tokens*. In thinking
    mode that usually means the reasoning consumed the budget and the answer is
    cut short, which is worth reporting plainly — the truncated JSON that follows
    would otherwise surface as a parse error that blames the model for garbage it
    never produced.
    """

    content: str
    reasoning: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() == "length"

    @property
    def reasoning_tokens(self) -> int:
        """Tokens spent thinking, as the API breaks them out in ``usage``."""
        details = self.usage.get("completion_tokens_details") or {}
        try:
            return int(details.get("reasoning_tokens") or 0)
        except (TypeError, ValueError):
            return 0


@dataclass
class DeepSeekConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int
    max_retries: int
    reasoning_effort: Optional[str] = None
    thinking: Optional[str] = None
    show_thinking: bool = False


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Find a DeepSeek API key: explicit arg -> env vars -> user config file."""
    if explicit:
        return explicit.strip()
    for var in config.DEEPSEEK_KEY_ENV_VARS:
        value = os.getenv(var)
        if value and value.strip():
            return value.strip()
    cfg = read_json(config.config_file_path())
    if isinstance(cfg, dict):
        value = cfg.get("deepseek_api_key")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


class DeepSeekClient:
    """Thin wrapper around the DeepSeek chat-completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking: Optional[str] = None,
        show_thinking: bool = False,
    ) -> None:
        key = resolve_api_key(api_key)
        self.cfg = DeepSeekConfig(
            api_key=key or "",
            base_url=base_url or config.DEEPSEEK_DEFAULT_BASE_URL,
            model=model or config.DEEPSEEK_DEFAULT_MODEL,
            timeout=timeout or config.DEEPSEEK_TIMEOUT,
            max_retries=max_retries if max_retries is not None else config.DEEPSEEK_MAX_RETRIES,
            reasoning_effort=(reasoning_effort or config.DEEPSEEK_DEFAULT_REASONING_EFFORT or None),
            thinking=(thinking or config.DEEPSEEK_DEFAULT_THINKING or None),
            show_thinking=show_thinking,
        )
        self._session = None
        #: The most recent :class:`ChatResult`. Kept so a caller that only wants
        #: the parsed JSON (via :meth:`chat_json`) can still reach the reasoning
        #: and token accounting without threading a return value through.
        self.last_result: Optional[ChatResult] = None

    # -- availability ---------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.cfg.api_key)

    def require(self) -> None:
        if not self.configured:
            raise DeepSeekError(
                "No DeepSeek API key found. Set DEEPSEEK_API_KEY (or pass --deepseek-api-key, "
                "or add \"deepseek_api_key\" to " + config.config_file_path() + ")."
            )

    # -- transport ------------------------------------------------------------

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = _endpoint(self.cfg.base_url)

        last_error: Optional[str] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                import requests  # type: ignore

                if self._session is None:
                    self._session = requests.Session()
                response = self._session.post(url, headers=headers, json=payload, timeout=self.cfg.timeout)
                status = response.status_code
                body = response.text
            except ImportError:  # pragma: no cover - urllib fallback
                status, body = self._post_urllib(url, headers, payload)
            except Exception as exc:  # network exception
                last_error = f"{type(exc).__name__}: {exc}"
                debug(f"DeepSeek request error (attempt {attempt}): {last_error}")
                time.sleep(min(2 ** attempt, 20))
                continue

            if status == 200:
                try:
                    return json.loads(body)
                except json.JSONDecodeError as exc:
                    raise DeepSeekError(f"DeepSeek returned invalid JSON: {exc}") from exc

            if status in (429, 500, 502, 503, 504):
                last_error = f"HTTP {status}: {body[:300]}"
                wait = min(2 ** attempt, 30)
                warn(f"DeepSeek transient error (HTTP {status}); retrying in {wait}s")
                time.sleep(wait)
                continue

            # Non-retryable.
            raise DeepSeekError(f"DeepSeek API error HTTP {status}: {body[:500]}")

        raise DeepSeekError(f"DeepSeek API unavailable after {self.cfg.max_retries} attempts: {last_error}")

    def _post_urllib(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]):
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")

    # -- streaming ------------------------------------------------------------

    def _iter_stream_events(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Yield decoded SSE events from a streaming completion.

        Retries only until the first event arrives. Once reasoning has been
        streamed to the user, reconnecting would replay text they have already
        read, so a failure mid-stream is raised rather than retried.
        """
        try:
            import requests  # type: ignore
        except ImportError as exc:  # pragma: no cover - requests is a hard dependency
            raise DeepSeekError(
                "live thinking display needs the 'requests' package (pip install requests)"
            ) from exc

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        url = _endpoint(self.cfg.base_url)
        last_error: Optional[str] = None

        for attempt in range(1, self.cfg.max_retries + 1):
            started = False
            try:
                if self._session is None:
                    self._session = requests.Session()
                with self._session.post(url, headers=headers, json=payload,
                                        timeout=self.cfg.timeout, stream=True) as response:
                    if response.status_code != 200:
                        body = response.text
                        if response.status_code in (429, 500, 502, 503, 504):
                            last_error = f"HTTP {response.status_code}: {body[:300]}"
                            wait = min(2 ** attempt, 30)
                            warn(f"DeepSeek transient error (HTTP {response.status_code}); retrying in {wait}s")
                            time.sleep(wait)
                            continue
                        raise DeepSeekError(
                            f"DeepSeek API error HTTP {response.status_code}: {body[:500]}")

                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            debug(f"ignoring non-SSE line: {line[:120]}")
                            continue
                        body = line[5:].strip()
                        if body == "[DONE]":
                            return
                        try:
                            event = json.loads(body)
                        except json.JSONDecodeError:
                            debug(f"ignoring unparsable SSE payload: {body[:120]}")
                            continue
                        started = True
                        yield event
                    return
            except DeepSeekError:
                raise
            except Exception as exc:
                if started:
                    raise DeepSeekError(
                        f"DeepSeek stream failed after output began: {type(exc).__name__}: {exc}"
                    ) from exc
                last_error = f"{type(exc).__name__}: {exc}"
                debug(f"DeepSeek stream error (attempt {attempt}): {last_error}")
                time.sleep(min(2 ** attempt, 20))
                continue

        raise DeepSeekError(
            f"DeepSeek API unavailable after {self.cfg.max_retries} attempts: {last_error}")

    def _consume_stream(self, payload: Dict[str, Any], *, label: Optional[str] = None) -> ChatResult:
        """Read a streaming completion, echoing the reasoning as it arrives.

        Every delta carries *both* ``content`` and ``reasoning_content`` — one of
        them empty, or ``null`` — so each field is concatenated on its own. A
        parser that branches on which key is present throws the answer away: the
        deltas that carry answer text also carry a (null) reasoning field.
        """
        show = self.cfg.show_thinking
        reasoning: List[str] = []
        content: List[str] = []
        usage: Dict[str, Any] = {}
        finish_reason: Optional[str] = None
        header_shown = False

        for event in self._iter_stream_events(payload):
            choices = event.get("choices") or []
            if not choices:
                # Usage arrives in a trailing event that carries no choices.
                if event.get("usage"):
                    usage = event["usage"]
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            think = delta.get("reasoning_content") or ""
            piece = delta.get("content") or ""
            if think:
                reasoning.append(think)
                if show:
                    if not header_shown:
                        thinking_header(label, model=self.cfg.model)
                        header_shown = True
                    thinking_chunk(think)
            if piece:
                content.append(piece)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]

        if header_shown:
            thinking_end()

        return ChatResult(content="".join(content), reasoning="".join(reasoning),
                          usage=usage, finish_reason=finish_reason)

    # -- public API -----------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        json_mode: bool = False,
        stream: bool = False,
        label: Optional[str] = None,
        thinking_override: Optional[str] = None,
    ) -> ChatResult:
        """Run one completion and return it with its reasoning.

        ``stream=True`` requests SSE and echoes the reasoning to stdout as it is
        generated; ``label`` names the block in that header (the stage being run).
        ``thinking_override`` forces thinking on or off for this call alone, so a
        truncated JSON reply can be retried with the whole budget going to the
        answer (see :meth:`chat_json`).
        """
        self.require()
        thinking = thinking_override or self.cfg.thinking
        thinking_on = (thinking or "").lower() == "enabled"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        # ``reasoning_effort`` is a thinking-mode parameter; sending it alongside
        # thinking:disabled is meaningless at best and a 400 at worst.
        if self.cfg.reasoning_effort and thinking_on:
            payload["reasoning_effort"] = self.cfg.reasoning_effort
        if thinking:
            payload["thinking"] = {"type": thinking}
        # Thinking mode accepts but ignores temperature/presence/frequency penalties,
        # so only send temperature when thinking is off.
        if not thinking_on:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        info(f"querying DeepSeek ({self.cfg.model}"
             f"{', effort=' + self.cfg.reasoning_effort if thinking_on and self.cfg.reasoning_effort else ''}"
             f"{', thinking=' + thinking if thinking else ''}"
             f"{', stream' if stream else ''})"
             f" — {sum(len(m.get('content', '')) for m in messages)} chars of context")
        started = time.time()
        if stream:
            result = self._consume_stream(payload, label=label)
        else:
            result = self._parse_completion(self._post(payload))
            if self.cfg.show_thinking and result.reasoning:
                # The flag promises the reasoning on stdout whatever the
                # transport, so a buffered reply is printed as one block.
                thinking_header(label, model=self.cfg.model)
                thinking_chunk(result.reasoning)
                thinking_end()
        elapsed = time.time() - started
        debug(f"DeepSeek responded in {elapsed:.1f}s, tokens={result.usage}")
        self.last_result = result
        return result

    def _parse_completion(self, response: Dict[str, Any]) -> ChatResult:
        """Pull a completion apart from a non-streaming response body."""
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DeepSeekError(f"unexpected DeepSeek response shape: {str(response)[:300]}") from exc
        return ChatResult(
            content=content or "",
            # null when the reply carried no thinking, and a string otherwise.
            reasoning=message.get("reasoning_content") or "",
            usage=response.get("usage") or {},
            finish_reason=choice.get("finish_reason"),
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8000,
        stream: bool = False,
        label: Optional[str] = None,
        thinking_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run one completion and parse a JSON object out of it.

        A truncated reply is retried once with thinking disabled before it is
        given up on — see :meth:`_retry_without_thinking`.
        """
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                           json_mode=True, stream=stream, label=label,
                           thinking_override=thinking_override)
        first: Optional[ChatResult] = None
        if result.truncated and thinking_override is None and self._thinking_enabled():
            first = result
            result = self._retry_without_thinking(
                messages, temperature=temperature, max_tokens=max_tokens, stream=stream,
                label=label, first=result)
        if result.truncated:
            # Report the *first* attempt's thinking spend: it is the attempt that
            # shows how much of the budget the chain of thought took.
            reasoning_tokens = max(result.reasoning_tokens,
                                   first.reasoning_tokens if first else 0)
            raise DeepSeekTruncation(self._truncation_message(
                max_tokens, reasoning_tokens, retried_without_thinking=first is not None))
        return parse_json_response(result.content)

    def _thinking_enabled(self) -> bool:
        return (self.cfg.thinking or "").lower() == "enabled"

    def _retry_without_thinking(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
        label: Optional[str],
        first: ChatResult,
    ) -> ChatResult:
        """Re-run a truncated JSON request once with thinking turned off.

        Thinking and the answer share a single completion budget, so a JSON reply
        that ran out of room is almost always one that spent its allowance
        reasoning. Disabling thinking for the retry hands the whole budget to the
        JSON — the only thing this call is for. The first attempt's chain of
        thought is merged onto the result so the saved transcript stays complete.
        """
        warn(f"DeepSeek reply was cut off at max_tokens={max_tokens} "
             "(finish_reason=length); retrying this stage once with thinking disabled")
        try:
            retried = self.chat(messages, temperature=temperature, max_tokens=max_tokens,
                                json_mode=True, stream=stream, label=label,
                                thinking_override="disabled")
        except DeepSeekError:
            # The retry could not even run; keep the first attempt reachable so the
            # caller still sees its reasoning and token accounting.
            self.last_result = first
            raise
        retried.reasoning = first.reasoning + retried.reasoning
        self.last_result = retried
        return retried

    def _truncation_message(self, max_tokens: int, reasoning_tokens: int,
                            *, retried_without_thinking: bool = False) -> str:
        """Explain a truncation, naming levers that can actually change the outcome."""
        spent = (f" {reasoning_tokens} of those tokens went to thinking."
                 if reasoning_tokens else "")
        if retried_without_thinking:
            advice = ("Even with thinking disabled the answer did not fit, so raise "
                      "--deepseek-max-tokens")
        elif self._thinking_enabled():
            advice = ("Raise --deepseek-max-tokens, or run with --deepseek-thinking disabled "
                      "so the whole completion budget belongs to the answer")
            effort = (self.cfg.reasoning_effort or "").lower()
            if effort and effort != "low":
                advice += f" (or lower --deepseek-reasoning-effort below {self.cfg.reasoning_effort})"
        else:
            advice = "Raise --deepseek-max-tokens"
        return (f"DeepSeek reply was cut off at max_tokens={max_tokens} "
                f"(finish_reason=length), so the JSON is incomplete.{spent} {advice}.")


def parse_json_response(raw: str) -> Dict[str, Any]:
    """Parse a JSON object out of a model response, tolerating code fences/preamble."""
    text = (raw or "").strip()
    if text.startswith("\ufeff"):
        text = text[1:]
    if text.startswith("```"):
        # Strip a leading ```json fence and its closing fence.
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise DeepSeekError(f"could not parse DeepSeek JSON response: {exc}") from exc
    raise DeepSeekError("DeepSeek response contained no JSON object")
