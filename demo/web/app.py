import datetime
import builtins
import asyncio
import json
import os
import re
import threading
import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast, AsyncIterator

import numpy as np
import torch
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

import copy

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_API_MODE = os.getenv("OLLAMA_API_MODE", "chat")
DEFAULT_CFG_SCALE = 2.3
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_LATENCY_SEC = 0.8

# TOKENIZERS_PARALLELISM controls whether Hugging Face tokenizers can use
# parallel worker threads in the current process. When True, tokenizers may
# spawn threads for faster batch encoding; when False, they run single-threaded
# to avoid warnings or deadlocks in forked worker scenarios. It can be safe to
# set True in long-running, non-forking server processes (or when you manage
# multiprocessing carefully), but in web servers that may fork or use worker
# pools it is safer to keep it False to prevent the "process just got forked"
# warning and potential hangs.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp


def _ollama_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    return headers


async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                yield line[5:].strip()


async def stream_ollama_chat(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    stop_event: asyncio.Event,
) -> AsyncIterator[str]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "stream": True,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload, headers=_ollama_headers()) as response:
            response.raise_for_status()
            async for data in _iter_sse_data(response):
                if stop_event.is_set():
                    break
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content


async def stream_ollama_responses(
    prompt: str,
    system_prompt: Optional[str],
    model: str,
    stop_event: asyncio.Event,
) -> AsyncIterator[str]:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/responses"
    payload = {
        "model": model,
        "stream": True,
        "input": prompt,
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload, headers=_ollama_headers()) as response:
            response.raise_for_status()
            async for data in _iter_sse_data(response):
                if stop_event.is_set():
                    break
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type", "")
                if event_type == "response.output_text.delta":
                    delta = event.get("delta")
                    if delta:
                        yield delta
                    continue
                if "choices" in event:
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content


def _find_sentence_boundary(text: str) -> int:
    matches = list(re.finditer(r"[.!?](?=\\s|$)", text))
    if not matches:
        return -1
    return matches[-1].end()


def _split_on_whitespace(text: str, min_chars: int) -> Tuple[str, str]:
    if len(text) <= min_chars:
        return text, ""
    cut = text.rfind(" ", 0, max(min_chars, 1))
    if cut == -1:
        cut = min_chars
    return text[:cut].rstrip(), text[cut:].lstrip()


def _make_segment_stop_event(session_stop: threading.Event) -> threading.Event:
    segment_stop = threading.Event()
    if session_stop.is_set():
        segment_stop.set()
        return segment_stop

    def watch_session_stop() -> None:
        session_stop.wait()
        segment_stop.set()

    threading.Thread(target=watch_session_stop, daemon=True).start()
    return segment_stop

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.rglob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-WHTest_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")
    
    service = StreamingTTSService(
        model_path=model_path,
        device=device
    )
    service.load()

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()
    print("[startup] Model ready.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)

@app.websocket("/stream_llm")
async def websocket_llm_stream(ws: WebSocket) -> None:
    await ws.accept()

    try:
        raw_start = await ws.receive_text()
        start_payload = json.loads(raw_start)
    except Exception:
        await ws.close(code=1003, reason="Invalid start payload")
        return

    prompt = (start_payload.get("prompt") or "").strip()
    system_prompt = (start_payload.get("system_prompt") or "").strip() or None
    model_name = (start_payload.get("model") or DEFAULT_OLLAMA_MODEL).strip()
    api_mode = (start_payload.get("api_mode") or DEFAULT_API_MODE).strip().lower()
    if api_mode not in ("chat", "responses"):
        api_mode = DEFAULT_API_MODE
    voice_param = start_payload.get("voice") or None

    cfg_scale = start_payload.get("cfg", DEFAULT_CFG_SCALE)
    steps_param = start_payload.get("steps")
    min_chars = start_payload.get("min_chars", DEFAULT_MIN_CHARS)
    max_latency_sec = start_payload.get("max_latency_sec", DEFAULT_MAX_LATENCY_SEC)

    try:
        cfg_scale = float(cfg_scale)
    except (TypeError, ValueError):
        cfg_scale = DEFAULT_CFG_SCALE
    if cfg_scale <= 0:
        cfg_scale = DEFAULT_CFG_SCALE
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except (TypeError, ValueError):
        inference_steps = None
    try:
        min_chars = max(20, int(min_chars))
    except (TypeError, ValueError):
        min_chars = DEFAULT_MIN_CHARS
    try:
        max_latency_sec = float(max_latency_sec)
    except (TypeError, ValueError):
        max_latency_sec = DEFAULT_MAX_LATENCY_SEC

    if not prompt:
        await ws.send_text(json.dumps({
            "type": "log",
            "event": "backend_error",
            "data": {"message": "Prompt is empty."},
            "timestamp": get_timestamp(),
        }))
        await ws.close(code=1003, reason="Empty prompt")
        return

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "llm_request_received",
            prompt_length=len(prompt),
            model=model_name,
            api_mode=api_mode,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
        )

        stop_event = asyncio.Event()
        session_stop_event = threading.Event()

        delta_queue: asyncio.Queue[str] = asyncio.Queue()
        llm_done = asyncio.Event()
        llm_error: Optional[Exception] = None

        async def llm_reader() -> None:
            nonlocal llm_error
            try:
                if api_mode == "responses":
                    stream = stream_ollama_responses(prompt, system_prompt, model_name, stop_event)
                else:
                    stream = stream_ollama_chat(prompt, system_prompt, model_name, stop_event)
                async for delta in stream:
                    if stop_event.is_set():
                        break
                    await delta_queue.put(delta)
                    enqueue_log("llm_delta", text=delta)
            except Exception as exc:
                llm_error = exc
            finally:
                llm_done.set()

        llm_task = asyncio.create_task(llm_reader())
        first_ws_send_logged = False

        async def stream_audio_segment(segment: str) -> None:
            nonlocal first_ws_send_logged
            segment_stop_event = _make_segment_stop_event(session_stop_event)
            iterator = streaming_tts(
                segment,
                cfg_scale=cfg_scale,
                inference_steps=inference_steps,
                voice_key=voice_param,
                stop_event=segment_stop_event,
                log_callback=enqueue_log,
            )
            sentinel = object()
            await flush_logs()
            try:
                while ws.client_state == WebSocketState.CONNECTED:
                    chunk = await asyncio.to_thread(next, iterator, sentinel)
                    if chunk is sentinel:
                        break
                    payload = service.chunk_to_pcm16(cast(np.ndarray, chunk))
                    await ws.send_bytes(payload)
                    if not first_ws_send_logged:
                        first_ws_send_logged = True
                        enqueue_log("backend_first_chunk_sent")
                    await flush_logs()
            finally:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()

        buffer = ""
        last_flush = time.monotonic()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                try:
                    delta = await asyncio.wait_for(delta_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    delta = ""

                if delta:
                    buffer += delta
                    await flush_logs()

                force_flush = (time.monotonic() - last_flush) >= max_latency_sec
                boundary_idx = _find_sentence_boundary(buffer)

                if boundary_idx != -1 and len(buffer) >= min_chars:
                    segment = buffer[:boundary_idx]
                    buffer = buffer[boundary_idx:].lstrip()
                elif force_flush and buffer:
                    if boundary_idx != -1:
                        segment = buffer[:boundary_idx]
                        buffer = buffer[boundary_idx:].lstrip()
                    else:
                        segment, buffer = _split_on_whitespace(buffer, min_chars)
                else:
                    segment = ""

                if segment:
                    enqueue_log("tts_segment_start", chars=len(segment))
                    await stream_audio_segment(segment)
                    enqueue_log("tts_segment_complete", chars=len(segment))
                    last_flush = time.monotonic()
                    await flush_logs()

                if llm_done.is_set() and delta_queue.empty():
                    break

            if buffer.strip():
                enqueue_log("tts_segment_start", chars=len(buffer))
                await stream_audio_segment(buffer)
                enqueue_log("tts_segment_complete", chars=len(buffer))
                await flush_logs()

            if llm_error:
                enqueue_log("llm_error", message=str(llm_error))
        except WebSocketDisconnect:
            enqueue_log("client_disconnected")
        finally:
            stop_event.set()
            session_stop_event.set()
            llm_task.cancel()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                await llm_task
            except Exception:
                pass
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
    finally:
        if acquired:
            lock.release()

@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else DEFAULT_CFG_SCALE
    except ValueError:
        cfg_scale = DEFAULT_CFG_SCALE
    if cfg_scale <= 0:
        cfg_scale = DEFAULT_CFG_SCALE
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
        )

        stop_signal = threading.Event()

        iterator = streaming_tts(
            text,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice_key=voice_param,
            log_callback=enqueue_log,
            stop_event=stop_signal,
        )
        sentinel = object()
        first_ws_send_logged = False

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                await flush_logs()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        finally:
            stop_signal.set()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
    }
