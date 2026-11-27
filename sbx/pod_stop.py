#!/usr/bin/env python3
"""Pod-side hard stop for ONE RunPod pod.  Start it at T+0, BEFORE the experiment, under `setsid nohup ... &`.

  python3 -m sbx.pod_stop --prefix <pod-name-prefix>- --cap-minutes 900 --save /workspace/sandbagging/results \\
          [--done-file /root/DONE] [--grace-minutes 5]

The pod stops at whichever comes first:
  * the hard cap: --cap-minutes after this script starts, whatever is running;
  * the end of the queue: once --done-file has existed for --grace-minutes.  The queue creates it as its LAST act.

It needs no secret of the operator's.  It uses only `runpodctl` and the RUNPOD_POD_ID that RunPod puts in the container's environment, and
it addresses that one pod and nothing else.  It refuses to arm (exit 2, nothing is ever stopped) unless RUNPOD_POD_ID is set, `runpodctl`
is on the PATH, a --prefix was given, the pod's name starts with that prefix, and the rescue location is on the volume.  There is no default
prefix, so it cannot match a pod it was not armed for.  Every decision is printed and appended to --log.

What a stop costs and keeps (RunPod's pod-management and storage documentation, read 2026-11-26): the GPU is released and compute billing
ends; the volume disk mounted at /workspace is KEPT and keeps billing for storage (cents an hour at pod sizes); the container disk is
CLEARED.  So a stopped pod is not a finished pod.  Results written under /workspace survive a stop; anything elsewhere on the pod does not,
and the pod has to be terminated by the operator once the results are off it.  Two consequences are built in:

  * --save DIR (repeatable): right before ANY stop, each directory is archived to <save-to>/<basename>.tgz.  A stop that loses the results
    has failed even when the pod stops.
  * --done-file is refused if it is on the volume: a done-file that survives a stop would stop the restarted pod again after one grace period.
    Its default is therefore /root/DONE, on the container disk.  (--log lives under /workspace on purpose: the stop log should outlive the pod.)

Whether `runpodctl` is present in a given image, and whether it stops the pod it runs in without an API key, is NOT assumed here.  It is
tested on a throwaway pod before any experiment pod is launched (docs/plan_sbx3.md, "Stop mechanism, before anything else"; that
planning document is not published).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time


def save_results(dirs, dest, log, excludes=("--exclude=*/cache", "--exclude=*/texts", "--exclude=*/arrays.npz")):
    """Archive each existing directory in `dirs` to <dest>/<basename>.tgz.  Returns the directories that could NOT be archived.

    Raw model traces and the packed array files are left out on purpose: the per-question scores (cells.jsonl, arrays/, qids/) are what
    the analysis needs, and a rescue copy must finish quickly on a disk that may be nearly full.  Nothing here raises: a rescue that
    fails is reported and the caller decides."""
    fails = []
    os.makedirs(dest, exist_ok=True)
    for d in dirs:
        if not d:
            continue
        d = d.rstrip("/")
        if not os.path.isdir(d):
            log(f"save: {d} does not exist, nothing to archive")
            continue
        parent, base = os.path.split(d)
        out = os.path.join(dest, base + ".tgz")
        tmp = out + ".part"
        try:
            if os.path.abspath(d) == os.path.abspath(dest) or os.path.abspath(dest).startswith(os.path.abspath(d) + os.sep):
                raise ValueError("the rescue location is inside the directory being archived")
            r = subprocess.run(["tar", "czf", tmp, *excludes, "-C", parent, base], capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                raise RuntimeError(f"tar exit {r.returncode}: {(r.stderr or '').strip()[:200]}")
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("the archive is empty")
            os.replace(tmp, out)                                   # atomic: a half-written archive never carries a finished name
            log(f"save: {d} -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            log(f"save FAILED for {d}: {type(e).__name__}: {e}")
            fails.append(d)
    return fails


class Log:
    def __init__(self, path=None):
        self.path = path
        self.lock = threading.Lock()

    def __call__(self, msg):
        line = f"[stop {time.strftime('%H:%M:%S')}] {msg}"
        with self.lock:
            print(line, flush=True)
            if self.path:
                try:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except OSError:
                    pass


def stop_pod(pod_id, log, runner=None):
    """Stop THIS pod with runpodctl and report the outcome.  Returns True iff runpodctl exited 0."""
    run = runner or subprocess.run
    try:
        r = run(["runpodctl", "stop", "pod", pod_id], capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        log("runpodctl is not on the PATH: the pod could not be stopped from inside")
        return False
    except subprocess.TimeoutExpired:
        log("runpodctl stop timed out after 180 s")
        return False
    out = (r.stdout or "").strip().replace("\n", " ")[:200]
    err = (r.stderr or "").strip().replace("\n", " ")[:200]
    log(f"runpodctl stop pod <this pod>: exit {r.returncode}; stdout: {out or '-'}; stderr: {err or '-'}")
    return r.returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=os.environ.get("NAME", ""), help="the pod's name (default: $NAME)")
    ap.add_argument("--prefix", required=True, help="the pod-name prefix this run is armed for, WITH its trailing dash; --name must start with it")
    ap.add_argument("--cap-minutes", type=float, required=True, help="hard cap measured from the start of this script")
    ap.add_argument("--save", action="append", default=[], help="a results directory to archive before any stop (repeatable)")
    ap.add_argument("--save-to", default="/workspace/rescue", help="where the rescue archives go; on the volume (/workspace), not the container disk")
    ap.add_argument("--done-file", default="/root/DONE", help="created by the queue as its last act; must NOT be on the volume "
                    "(a done-file that survives a stop would stop the restarted pod again)")
    ap.add_argument("--grace-minutes", type=float, default=5.0)
    ap.add_argument("--poll-seconds", type=float, default=10.0)
    ap.add_argument("--log", default="/workspace/logs/stop.log")
    ap.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""), help="default: $RUNPOD_POD_ID")
    ap.add_argument("--dry-run", action="store_true",
                    help="every refusal and the rescue archive as in a real run, but never call runpodctl; only this flag excuses a missing runpodctl")
    ap.add_argument("--volume", default="/workspace", help="the volume mount point of this pod (RunPod: /workspace)")
    a = ap.parse_args(argv)
    log = Log(None if a.dry_run else a.log)

    pod_id = a.pod_id
    on_volume = lambda p: os.path.abspath(p) == a.volume or os.path.abspath(p).startswith(a.volume.rstrip("/") + os.sep)   # noqa: E731
    # ---- refusals: nothing below this block can stop anything
    if not pod_id:
        log("REFUSING: RUNPOD_POD_ID is not set, so this is not a pod or I cannot tell which one. Nothing will be stopped.")
        return 2
    if not a.prefix or not a.prefix.endswith("-") or " " in a.prefix:
        log(f"REFUSING: --prefix {a.prefix!r} is empty, has no trailing dash or contains a space. Nothing will be stopped.")
        return 2
    if not a.name.startswith(a.prefix) or " " in a.name or len(a.name) <= len(a.prefix):
        log(f"REFUSING: pod name {a.name!r} does not start with {a.prefix!r} (not one this run was armed for). Nothing will be stopped.")
        return 2
    if shutil.which("runpodctl") is None and not a.dry_run:
        log("REFUSING: runpodctl is not installed, so nothing here could stop this pod. Nothing will be stopped.")
        return 2
    if not on_volume(a.save_to):
        log(f"REFUSING: --save-to {a.save_to!r} is not on the volume ({a.volume}), so the rescue archive would be wiped by the stop it precedes. "
            "Nothing will be stopped.")
        return 2
    if on_volume(a.done_file):
        log(f"REFUSING: --done-file {a.done_file!r} is on the volume, so it would survive a stop and stop the restarted pod again. "
            "Nothing will be stopped.")
        return 2
    log(f"armed for {a.name} (pod id {pod_id[:4]}...): hard cap {a.cap_minutes:g} min; rescue {a.save or 'nothing'} -> {a.save_to}; "
        f"done-file {a.done_file} then {a.grace_minutes:g} min grace" + ("   [DRY RUN: runpodctl will not be called]" if a.dry_run else ""))

    lock = threading.Lock()
    state = ["new"]                           # under `lock`: "new" -> "stopping" -> "stopped", or back to "failed"
    finished = threading.Event()              # a stop SUCCEEDED (or was withheld in a dry run): the process may end

    def stop(reason):
        """Rescue the results, then stop.  One function for both paths; the lock and `state` decide what a second arrival does:
          a stop is under way, or one has succeeded -> stand down (a stop is never issued twice);
          the earlier stop FAILED               -> try again (that is what the hard cap is for)."""
        with lock:
            if state[0] in ("stopping", "stopped"):
                log(f"{reason}: a stop is already under way or has succeeded; standing down")
                return True
            log(f"{reason}: saving results first")
            fails = save_results(a.save, a.save_to, log)
            if fails:
                log(f"WARNING: {len(fails)} of {len(a.save)} rescue archive(s) failed; stopping anyway, the cap is the operator's limit")
            state[0] = "stopping"
            if a.dry_run:
                log("DRY RUN: this is where `runpodctl stop pod <this pod>` would be issued; nothing was stopped")
                state[0] = "stopped"
                finished.set()
                return True
            ok = stop_pod(pod_id, log)
            state[0] = "stopped" if ok else "failed"
            if ok:
                finished.set()
            else:
                log("the stop FAILED; if the hard cap has not been reached yet it will be retried then")
            return ok

    def cap():
        """The hard cap.  It waits for the cap time unless a stop has SUCCEEDED; nothing else releases it, so a queue-path stop that
        hangs or fails can never silence it."""
        finished.wait(a.cap_minutes * 60)
        if finished.is_set():
            return                            # a stop succeeded before the cap: nothing left to do
        stop(f"HARD CAP reached ({a.cap_minutes:g} min): stopping the pod")
        log("the hard cap stop has been attempted; the container disk is cleared by a stop, so what was not rescued to the volume or collected is lost")

    cap_thread = threading.Thread(target=cap, name="hard-cap", daemon=True)
    cap_thread.start()

    # ---- the end of the queue
    t0 = time.time()
    seen = None
    while time.time() - t0 < a.cap_minutes * 60:
        if os.path.exists(a.done_file):
            if seen is None:
                seen = time.time()
                log(f"done-file present: the queue says the results are safe; stopping in {a.grace_minutes:g} min unless it is removed")
            elif time.time() - seen >= a.grace_minutes * 60:
                stop("stopping the pod (the queue finished)")     # the outcome is read from `state` below, never by returning here:
                break                                               # this process must outlive a failed stop, so the cap can retry it
        elif seen is not None:
            log("done-file removed again: the queue is still working; grace cancelled")
            seen = None
        time.sleep(min(a.poll_seconds, max(0.05, a.cap_minutes * 60 - (time.time() - t0))))
    # Only a SUCCESSFUL stop (or a dry run) ends this process early.  Otherwise it waits for the cap thread, which retries a failed stop
    # or stands down beside one that is still under way.  The cap thread is a daemon: if this process returned while it was still
    # working, the stop it is issuing could be destroyed before it is sent.
    finished.wait(a.cap_minutes * 60 + 240)                         # + stop_pod's own 180 s timeout, with room
    cap_thread.join(timeout=240)
    ok = state[0] == "stopped"
    log("finished: the pod " + ("was stopped" if ok else "COULD NOT BE STOPPED from inside; it must be stopped from the operator's side") + ". "
        "The container disk is cleared by a stop; the pod must still be terminated once the results are off it.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
