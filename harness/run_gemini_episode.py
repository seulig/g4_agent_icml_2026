#!/usr/bin/env python3
"""
Gemini agentic episode runner for ICML v4 cross-model validation.

Usage:
    export GOOGLE_API_KEY="your-key"
    python3 run_gemini_episode.py ep_NNN [--model gemini-2.5-pro]

The script:
1. Reads PROMPT.txt from the episode directory
2. Gives Gemini tool-use access to bash and file read/write
3. Runs an agentic loop until Gemini signals completion
4. Saves the full transcript to transcript_gemini.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

# ---------- Configuration ----------

BASE = None  # set from --dir argument
DEFAULT_MODEL = "gemini-2.5-flash"
GCP_PROJECT = "icml2026-ws"
GCP_LOCATION = "us-central1"
MAX_TURNS = 500  # safety limit
BASH_TIMEOUT = 15000  # seconds per command (electron sims at d=10m E=40 GeV can take ~3.5 hours)

# ---------- Tool definitions ----------

bash_tool = types.FunctionDeclaration(
    name="bash",
    description="Execute a bash command and return stdout+stderr. Use for running simulations, python scripts, installing packages, etc. The working directory persists between calls.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "command": types.Schema(
                type="STRING",
                description="The bash command to execute."
            ),
        },
        required=["command"],
    ),
)

read_file_tool = types.FunctionDeclaration(
    name="read_file",
    description="Read the contents of a file and return it as text.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "path": types.Schema(
                type="STRING",
                description="Absolute path to the file to read."
            ),
        },
        required=["path"],
    ),
)

write_file_tool = types.FunctionDeclaration(
    name="write_file",
    description="Write content to a file, creating it if it doesn't exist or overwriting if it does.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "path": types.Schema(
                type="STRING",
                description="Absolute path to the file to write."
            ),
            "content": types.Schema(
                type="STRING",
                description="The content to write to the file."
            ),
        },
        required=["path", "content"],
    ),
)

done_tool = types.FunctionDeclaration(
    name="done",
    description="Signal that you have completed the task. Call this when you have finished all analysis and saved your results.",
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "summary": types.Schema(
                type="STRING",
                description="Brief summary of what was accomplished."
            ),
        },
        required=["summary"],
    ),
)

tools = types.Tool(function_declarations=[bash_tool, read_file_tool, write_file_tool, done_tool])

# ---------- Tool execution ----------

cwd = None  # will be set to episode directory
episode_dir = None  # stable reference to episode directory (doesn't change with cd)


def execute_bash(command: str) -> str:
    global cwd
    try:
        # Wrap command to handle cd persistence: prepend cd to current cwd,
        # and capture any cd at the end of the command to update cwd
        wrapped = f"cd {cwd} && {command} && pwd > /tmp/_gemini_cwd"
        result = subprocess.run(
            wrapped,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            cwd=cwd,
        )
        # Update cwd if the command included a cd
        try:
            new_cwd = Path("/tmp/_gemini_cwd").read_text().strip()
            if new_cwd and Path(new_cwd).is_dir():
                cwd = new_cwd
        except Exception:
            pass

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if result.stdout else result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"

        # For OMSim simulations: replace the massive Geant4 log with a concise success report.
        # The agent doesn't need the G4Material warnings — it just needs to know whether the
        # simulation succeeded and what the output file contains.
        if "OMSim_WavePID_study" in command and "-o " in command:
            try:
                import re
                m = re.search(r'-o\s+(\S+)', command)
                if m:
                    raw_path = m.group(1).strip('"').strip("'")
                    out_path = raw_path + "_hits.root"
                    # If path contains unexpanded shell variables, scan episode dir instead
                    if '$' in out_path or not Path(out_path).is_absolute():
                        import glob
                        recent = sorted(Path(episode_dir).glob('*_hits.root'), key=lambda p: p.stat().st_mtime, reverse=True)
                        if recent:
                            out_path = str(recent[0])
                        else:
                            return f"[SIMULATION COMPLETE but could not locate output file in {episode_dir}]"
                    if Path(out_path).exists():
                        size = Path(out_path).stat().st_size
                        if size > 1000:
                            # Try to read event/hit counts
                            try:
                                import uproot
                                f = uproot.open(out_path)
                                if 'PhotonHits' in f:
                                    tree = f['PhotonHits']
                                    ev = tree['eventID'].array(library='np')
                                    n_events = len(set(ev))
                                    n_hits = len(ev)
                                    comp_time = ""
                                    tm = re.search(r'Computation time:\s+([\d.]+)\s+seconds', output)
                                    if tm:
                                        comp_time = f", compute time {float(tm.group(1)):.0f}s"
                                    return (f"[SIMULATION SUCCESS: {out_path}]\n"
                                            f"  Events with hits: {n_events}, Total photon hits: {n_hits}{comp_time}")
                            except Exception:
                                pass
                            return f"[SIMULATION SUCCESS: {out_path} ({size} bytes)]"
                        else:
                            return f"[SIMULATION FAILED: output file is empty ({size} bytes) - the run was likely killed. Try again with fewer events.]"
                    else:
                        return f"[SIMULATION FAILED: output file not created at {out_path}]"
            except Exception as e:
                return f"[Error interpreting simulation result: {e}]\n\n{output[:2000]}"

        # For non-OMSim commands: keep head+tail truncation for long outputs
        if len(output) > 10000:
            head = output[:3000]
            tail = output[-5000:]
            output = head + f"\n... [TRUNCATED {len(output) - 8000} chars] ...\n" + tail

        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT after {BASH_TIMEOUT}s]"
    except Exception as e:
        return f"[ERROR: {e}]"


def execute_read_file(path: str) -> str:
    try:
        p = Path(path)
        if not p.is_absolute():
            p = Path(episode_dir) / p
        content = p.read_text()
        if len(content) > 100000:
            return content[:100000] + f"\n... [truncated, {len(content)} chars total]"
        return content if content else "(empty file)"
    except Exception as e:
        return f"[ERROR reading {path}: {e}]"


def execute_write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        # Resolve relative paths against episode directory, not cwd
        if not p.is_absolute():
            p = Path(episode_dir) / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} chars to {p}"
    except Exception as e:
        return f"[ERROR writing {path}: {e}]"


def handle_function_call(fn_name: str, fn_args: dict) -> str:
    if fn_name == "bash":
        return execute_bash(fn_args["command"])
    elif fn_name == "read_file":
        return execute_read_file(fn_args["path"])
    elif fn_name == "write_file":
        return execute_write_file(fn_args["path"], fn_args["content"])
    elif fn_name == "done":
        return "DONE: " + fn_args.get("summary", "")
    else:
        return f"[Unknown tool: {fn_name}]"


# ---------- Main loop ----------

def run_episode(ep_dir_name: str, model_name: str):
    global cwd, episode_dir

    ep_path = BASE / ep_dir_name
    episode_dir = str(ep_path)
    prompt_path = ep_path / "PROMPT.txt"

    if not prompt_path.exists():
        print(f"ERROR: {prompt_path} not found")
        sys.exit(1)

    cwd = str(ep_path)
    prompt_text = prompt_path.read_text()

    # System instruction for isolation
    system_instruction = (
        "You are a scientific research agent with access to bash, file read/write tools. "
        "You are investigating a particle physics simulation. "
        "Do not access any files outside your episode directory and the simulation binary directory. "
        "Do not browse parent or sibling directories. "
        "Run your own simulations from scratch. "
        "IMPORTANT: Simulations can take 5-20 minutes each. If a simulation command "
        "seems to hang, wait — it is running. Do NOT assume it has failed until you "
        "see a timeout error. If you do hit a timeout, reduce -n from 50 to 20 events. "
        "When you are done with all analysis, call the 'done' tool with a summary. "
        f"Your working directory is: {ep_path}"
    )

    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tools],
        temperature=0.7,
    )

    # Conversation history
    transcript = []
    contents = [types.Content(role="user", parts=[types.Part(text=prompt_text)])]
    transcript.append({"role": "user", "text": prompt_text[:500] + "..."})

    print(f"=== Starting Gemini episode: {episode_dir} ===")
    print(f"Model: {model_name}")
    print(f"Directory: {ep_path}")
    print()

    for turn in range(MAX_TURNS):
        print(f"--- Turn {turn + 1}/{MAX_TURNS} ---")

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            print(f"API ERROR: {e}")
            transcript.append({"role": "error", "text": str(e)})
            time.sleep(5)
            continue

        # Process the response
        if not response.candidates or not response.candidates[0].content.parts:
            print("Empty response, ending.")
            break

        assistant_parts = response.candidates[0].content.parts
        contents.append(types.Content(role="model", parts=assistant_parts))

        # Check for text output
        for part in assistant_parts:
            if hasattr(part, "text") and part.text:
                print(f"GEMINI: {part.text[:300]}{'...' if len(part.text) > 300 else ''}")
                transcript.append({"role": "assistant", "text": part.text})

        # Check for function calls
        function_calls = [p for p in assistant_parts if hasattr(p, "function_call") and p.function_call]

        if not function_calls:
            # Agent returned text without a tool call. Prompt it to continue
            # rather than ending — it may just be writing intermediate reasoning.
            # Only break if this happens 3 times in a row (agent truly done).
            no_tool_count = getattr(run_episode, '_no_tool_count', 0) + 1
            run_episode._no_tool_count = no_tool_count
            if no_tool_count >= 10:
                print("No tool calls 10x in a row, ending.")
                break
            print("No tool calls, prompting to continue...")
            contents.append(types.Content(role="user", parts=[
                types.Part(text="You must use tools to proceed. Write your analysis code using write_file, then run it with bash. Do not just describe what you would do — actually do it using tools. When completely finished with all steps, call the done tool with your summary.")
            ]))
            continue

        # Reset no-tool counter since we got tool calls
        run_episode._no_tool_count = 0

        # Execute all function calls
        function_responses = []
        done = False
        for part in function_calls:
            fc = part.function_call
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}

            print(f"  TOOL: {fn_name}({json.dumps(fn_args)[:200]})")
            result = handle_function_call(fn_name, fn_args)
            print(f"  RESULT: {result[:200]}{'...' if len(result) > 200 else ''}")

            transcript.append({
                "role": "tool_call",
                "name": fn_name,
                "args": fn_args,
                "result": result[:5000],
            })

            function_responses.append(
                types.Part(function_response=types.FunctionResponse(
                    name=fn_name,
                    response={"result": result[:10000]},
                ))
            )

            if fn_name == "done":
                done = True

        contents.append(types.Content(role="user", parts=function_responses))

        if done:
            print("\n=== Agent signaled DONE ===")
            break

    else:
        print(f"\n=== Reached max turns ({MAX_TURNS}) ===")

    # Save transcript
    transcript_path = ep_path / "transcript_gemini.json"
    with open(transcript_path, "w") as f:
        json.dump(transcript, f, indent=2, default=str)
    print(f"\nTranscript saved to {transcript_path}")
    print(f"Total turns: {turn + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Gemini episode")
    parser.add_argument("episode", help="Episode directory name (e.g., ep_001)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dir", default="/Users/steveneulig/Desktop/icml_v4/episodes_gemini",
                        help="Base episodes directory")
    args = parser.parse_args()

    BASE = Path(args.dir)
    run_episode(args.episode, args.model)
