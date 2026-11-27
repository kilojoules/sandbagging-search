"""sbx.pod_stop: the pod-side hard stop, exercised with a fake runpodctl and timings of a few seconds (no pod, no network).

The volume is a stand-in directory (--volume) so the "must be on the volume / must not be on the volume" rules can be tested off a pod.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tarfile
import threading
import time

from sbx import pod_stop

FAKE = '''#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps({"t": time.time(), "argv": sys.argv[1:]}) + "\\n")
print("ok")
'''
SCRIPT = os.path.join(os.path.dirname(pod_stop.__file__), "pod_stop.py")
PREFIX = "owner-sbx-"                                   # stands for whatever prefix a run is armed for


def _fake_runpodctl(tmp_path):
    d = tmp_path / "bin"
    d.mkdir()
    f = d / "runpodctl"
    f.write_text(FAKE)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(d), str(tmp_path / "calls.jsonl")


def _calls(log):
    if not os.path.exists(log):
        return []
    return [json.loads(l)["argv"] for l in open(log, encoding="utf-8") if l.strip()]


def _args(tmp_path, name=PREFIX + "test", cap_min=0.06, done=None, grace_min=0.03, pod_id="pod-aaaa1111", poll=0.05, extra=()):
    """A valid arming against a stand-in volume at <tmp>/volume.  `extra` can override any option (argparse keeps the LAST value)."""
    vol = tmp_path / "volume"
    return ["--name", name, "--prefix", PREFIX, "--cap-minutes", str(cap_min), "--volume", str(vol),
            "--done-file", str(done if done else tmp_path / "logs" / "DONE"),                 # off the volume: the default situation
            "--save-to", str(vol / "rescue"),                                                # on the volume
            "--grace-minutes", str(grace_min), "--poll-seconds", str(poll), "--log", str(tmp_path / "stop.log"), "--pod-id", pod_id, *extra]


def _run(args, env_extra, limit):
    """run pod_stop.py as a subprocess for up to `limit` seconds; returns (exit code, seconds it ran, output)"""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, SCRIPT, *args], env=dict(os.environ, **env_extra), capture_output=True, text=True, timeout=limit)
        return p.returncode, time.time() - t0, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return "timeout", time.time() - t0, out


# ------------------------------------------------------------------ the two ways a pod gets stopped

def test_queue_finished_stops_once_after_grace(tmp_path):
    bindir, log = _fake_runpodctl(tmp_path)
    done = tmp_path / "logs" / "DONE"
    done.parent.mkdir()
    threading.Thread(target=lambda: (time.sleep(1.0), done.write_text(""))).start()       # the queue finishes after 1 s
    rc, secs, out = _run(_args(tmp_path, cap_min=0.20, done=done, grace_min=0.05),      # cap 12 s, grace 3 s
                         {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert rc == 0, (rc, out)
    assert "stopping the pod (the queue finished)" in out and "HARD CAP" not in out, out
    assert _calls(log) == [["stop", "pod", "pod-aaaa1111"]], _calls(log)                 # exactly one stop, this pod only
    assert 3.5 < secs < 9, secs                                                          # after the grace, well before the cap


def test_queue_never_finishes_cap_fires_once(tmp_path):
    bindir, log = _fake_runpodctl(tmp_path)
    rc, secs, out = _run(_args(tmp_path, cap_min=0.06), {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert "HARD CAP reached (0.06 min): stopping the pod" in out, out
    assert abs(secs - 3.6) < 2.5, secs                                                    # 0.06 min = 3.6 s
    time.sleep(0.5)
    assert _calls(log) == [["stop", "pod", "pod-aaaa1111"]], _calls(log)                 # exactly once, not once per thread


def test_done_file_removed_cancels_grace(tmp_path):
    bindir, log = _fake_runpodctl(tmp_path)
    done = tmp_path / "logs" / "DONE"
    done.parent.mkdir()

    def touch_and_remove():
        time.sleep(0.8)
        done.write_text("")
        time.sleep(0.8)
        done.unlink()

    threading.Thread(target=touch_and_remove).start()
    rc, secs, out = _run(_args(tmp_path, cap_min=0.07, done=done, grace_min=0.05),     # grace 3 s, cap 4.2 s
                         {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert "grace cancelled" in out and "queue finished" not in out, out                 # it did NOT stop on the grace path
    assert "HARD CAP reached" in out, out
    assert _calls(log) == [["stop", "pod", "pod-aaaa1111"]], _calls(log)                 # the cap fired later, once


# A fake runpodctl for the cases that only matter when the FIRST stop is slow or fails.  Every arrival is recorded before anything else.
#   FAIL_FIRST=1  the first call fails (exit 1); later calls succeed
#   HOLD_S=<sec>  each call sleeps that long before answering (a real `runpodctl stop` takes seconds)
SLOW_FAKE = '''#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["FAKE_LOG"], "a") as fh:
    fh.write(json.dumps({"t": time.time(), "argv": sys.argv[1:]}) + "\\n")
n = sum(1 for _ in open(os.environ["FAKE_LOG"]))
time.sleep(float(os.environ.get("HOLD_S", "0")))
if os.environ.get("FAIL_FIRST") == "1" and n == 1:
    print("boom", file=sys.stderr)
    sys.exit(1)
print("ok")
'''


def _slow_runpodctl(tmp_path, **env):
    d = tmp_path / "bin"
    d.mkdir()
    f = d / "runpodctl"
    f.write_text(SLOW_FAKE)
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return dict(os.environ, PATH=str(d) + os.pathsep + os.environ["PATH"], FAKE_LOG=str(tmp_path / "calls.jsonl"), **env)


def test_a_stop_still_running_when_the_cap_arrives_is_not_repeated(tmp_path):
    """The race the once-only rule exists for.  The queue path issues its stop and that stop is SLOW (a real one takes seconds); the
    hard cap arrives while it is still running.  Exactly one `runpodctl stop` may result."""
    env = _slow_runpodctl(tmp_path, HOLD_S="4")
    done = tmp_path / "logs" / "DONE"
    done.parent.mkdir()
    done.write_text("")                                              # the queue has already finished when the script starts
    cap_s = 2.0
    rc, secs, out = _run(_args(tmp_path, cap_min=cap_s / 60, done=done, grace_min=0.01, poll=0.05), env, 60)   # stop at ~0.6 s, cap at 2 s
    calls = _calls(env["FAKE_LOG"])
    assert len(calls) == 1 and calls[0] == ["stop", "pod", "pod-aaaa1111"], (calls, out)
    assert rc == 0 and "a stop is already under way or has succeeded; standing down" in out, (rc, out)


def test_a_failed_queue_stop_is_retried_by_the_hard_cap(tmp_path):
    """If the queue path's stop FAILS, nothing has been gained by staying silent: the hard cap must still act.  A stop in flight or
    failed may never release the cap thread early."""
    env = _slow_runpodctl(tmp_path, FAIL_FIRST="1")
    done = tmp_path / "logs" / "DONE"
    done.parent.mkdir()
    done.write_text("")
    cap_s = 3.0
    rc, secs, out = _run(_args(tmp_path, cap_min=cap_s / 60, done=done, grace_min=0.01, poll=0.05), env, 60)   # first stop at ~0.6 s
    calls = _calls(env["FAKE_LOG"])
    assert len(calls) == 2 and all(c == ["stop", "pod", "pod-aaaa1111"] for c in calls), (calls, out)         # failed once, retried once
    assert "the stop FAILED" in out and "HARD CAP reached" in out, out
    assert rc == 0, (rc, out)                                                          # the retry is what the process reports
    assert secs >= cap_s - 0.5, (secs, cap_s)                                           # and it waited for the cap to do it


def test_a_stop_that_hangs_is_not_waited_on_forever_by_the_loop(tmp_path):
    """The queue path's stop hangs past the cap.  The cap thread reaches the stop, finds one under way, and stands down; the process
    does not issue a second stop.  (A hung `runpodctl` is bounded by stop_pod's own 180 s timeout.)"""
    env = _slow_runpodctl(tmp_path, HOLD_S="5")
    done = tmp_path / "logs" / "DONE"
    done.parent.mkdir()
    done.write_text("")
    rc, secs, out = _run(_args(tmp_path, cap_min=0.03, done=done, grace_min=0.01, poll=0.05), env, 60)         # cap at 1.8 s
    assert len(_calls(env["FAKE_LOG"])) == 1, (_calls(env["FAKE_LOG"]), out)
    assert rc == 0, (rc, out)


def test_queue_finishing_just_before_the_cap_stops_once(tmp_path):
    """Timing version, no hold: the queue's grace ends a moment before the cap."""
    bindir, log = _fake_runpodctl(tmp_path)
    done = tmp_path / "logs" / "DONE"
    done.parent.mkdir()
    threading.Thread(target=lambda: (time.sleep(3.3), done.write_text(""))).start()    # grace ends at 5.1 s; the cap is at 3.6 s
    rc, secs, out = _run(_args(tmp_path, cap_min=0.06, done=done, grace_min=0.03),
                         {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert _calls(log) == [["stop", "pod", "pod-aaaa1111"]], (_calls(log), out)
    assert out.count("saving results first") == 1, out


# ------------------------------------------------------------------ refusals: nothing is ever stopped

def test_refuses_without_pod_id_or_with_a_foreign_name(tmp_path):
    bindir, log = _fake_runpodctl(tmp_path)
    env = {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}
    for kwargs in ({"pod_id": ""}, {"name": "someone-elses-pod"}, {"name": "owner-sbx-"}, {"name": "owner-other-1"},
                   {"name": PREFIX + "has space"}):
        rc, secs, out = _run(_args(tmp_path, **kwargs), env, 30)
        assert rc == 2 and "REFUSING" in out, (rc, kwargs, out)
    assert _calls(log) == [], _calls(log)                                                # runpodctl was never invoked


def test_the_prefix_must_be_given_and_must_end_in_a_dash(tmp_path):
    """No default prefix: a script that cannot say whose pod it stops must refuse rather than guess."""
    bindir, log = _fake_runpodctl(tmp_path)
    env = {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}
    base = _args(tmp_path)
    no_prefix = [x for i, x in enumerate(base) if i not in (base.index("--prefix"), base.index("--prefix") + 1)]
    rc, secs, out = _run(no_prefix, env, 30)                              # the flag omitted entirely: argparse itself refuses
    assert rc == 2 and "the following arguments are required: --prefix" in out, (rc, out)
    for args in ([x if x != PREFIX else "owner-sbx" for x in base],      # no trailing dash: would also match owner-sbxel-...
                 [x if x != PREFIX else "" for x in base]):
        rc, secs, out = _run(args, env, 30)
        assert rc == 2 and "REFUSING" in out, (rc, out)
    assert _calls(log) == [], _calls(log)                                 # nothing was stopped in any of the three


def test_refuses_a_rescue_location_that_a_stop_would_wipe(tmp_path):
    """The container disk is cleared by the stop, so an archive written there would not outlive the thing it protects."""
    bindir, log = _fake_runpodctl(tmp_path)
    env = {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}
    elsewhere = tmp_path / "container-disk" / "rescue"
    for where in (str(elsewhere), str(tmp_path / "volume2" / "rescue"), ""):
        rc, secs, out = _run(_args(tmp_path, extra=("--save-to", where)), env, 30)
        assert rc == 2 and "wiped by the stop it precedes" in out, (rc, where, out)
    assert _calls(log) == [], _calls(log)


def test_refuses_a_done_file_on_the_volume(tmp_path):
    """A done-file on the volume survives a stop.  A restarted pod would find it and stop again one grace period after booting."""
    bindir, log = _fake_runpodctl(tmp_path)
    env = {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}
    stale = tmp_path / "volume" / "DONE"
    rc, secs, out = _run(_args(tmp_path, extra=("--done-file", str(stale))), env, 30)
    assert rc == 2 and "would survive a stop and stop the restarted pod again" in out, (rc, out)
    assert _calls(log) == [], _calls(log)


def test_refuses_when_runpodctl_is_missing(tmp_path):
    empty = tmp_path / "emptybin"
    empty.mkdir()
    rc, secs, out = _run(_args(tmp_path), {"PATH": str(empty)}, 30)
    assert rc == 2 and "REFUSING" in out and "runpodctl is not installed" in out, out


def test_the_default_paths_are_consistent_with_each_other(tmp_path):
    """The defaults must be usable as they are.  /workspace is the volume: the rescue archive belongs there, the done-file must NOT be
    there.  A default the script itself refuses would only be found on a real pod."""
    d = tmp_path / "defaults"
    d.mkdir()
    bindir, log = _fake_runpodctl(d)
    fake_id = "pod-dddd4444"
    # the defaults that matter are RELATIVE to the volume: --save-to under it, --done-file off it.  The volume is moved to a stand-in and
    # the rescue location moves with it; the done-file default is left exactly as shipped, and must then pass the off-volume check.
    vol = tmp_path / "workspace"
    args = ["--name", PREFIX + "test", "--prefix", PREFIX, "--cap-minutes", "0.02", "--volume", str(vol),
            "--save-to", str(vol / "rescue"), "--log", str(d / "stop.log"), "--pod-id", fake_id]
    rc, secs, out = _run(args, {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert "REFUSING" not in out and "armed for" in out, out                      # the defaults arm
    assert f"done-file /root/DONE" in out, out                                     # and the default done-file is off the volume
    assert rc == 0 and _calls(log) == [["stop", "pod", fake_id]], (rc, _calls(log), out)


# ------------------------------------------------------------------ the rescue archive comes before any stop

def _results(tmp_path):
    r = tmp_path / "results" / "cedar"
    (r / "native" / "arrays").mkdir(parents=True)
    (r / "native" / "cells.jsonl").write_text('{"acc": 0.5}\n')
    (r / "native" / "arrays" / "a.npz").write_bytes(b"x" * 200)
    (r / "native" / "texts").mkdir()
    (r / "native" / "texts" / "t.jsonl").write_text("raw trace" * 1000)
    return r


def test_the_hard_cap_saves_the_results_to_the_volume_before_stopping(tmp_path):
    """The order is the point: archive first, then stop.  An unsaved result on the container disk is lost."""
    bindir, log = _fake_runpodctl(tmp_path)
    results = _results(tmp_path)
    rc, secs, out = _run(_args(tmp_path, cap_min=0.06, extra=("--save", str(results))),
                         {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert "HARD CAP reached" in out and "saving results first" in out, out
    arc = tmp_path / "volume" / "rescue" / "cedar.tgz"
    assert arc.exists() and arc.stat().st_size > 50, list((tmp_path / "volume").iterdir())
    with tarfile.open(arc) as tf:
        names = tf.getnames()
    assert "cedar/native/cells.jsonl" in names and "cedar/native/arrays/a.npz" in names, names
    assert not any("texts" in n for n in names), names                        # raw traces are left out on purpose
    assert _calls(log) == [["stop", "pod", "pod-aaaa1111"]]
    assert out.index("save: ") < out.index("runpodctl stop pod"), out          # the archive was written before the stop was issued


def test_a_failing_rescue_is_reported_and_the_pod_is_still_stopped(tmp_path):
    """A rescue that cannot be written must not leave the GPU billing.  It is logged, and the stop goes ahead."""
    bindir, log = _fake_runpodctl(tmp_path)
    broken = tmp_path / "results" / "gone"
    rc, secs, out = _run(_args(tmp_path, cap_min=0.06, extra=("--save", str(broken))),
                         {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}, 30)
    assert "gone does not exist, nothing to archive" in out, out
    assert "HARD CAP reached" in out and _calls(log) == [["stop", "pod", "pod-aaaa1111"]]


def test_rescue_reports_what_it_cannot_archive_instead_of_raising(tmp_path):
    seen = []
    dest = tmp_path / "volume" / "rescue"
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "cells.jsonl").write_text("{}\n")
    assert pod_stop.save_results([str(ok), str(tmp_path / "missing"), ""], str(dest), seen.append) == []
    assert (dest / "ok.tgz").exists() and any("does not exist" in m for m in seen)
    # the rescue location inside the directory being archived: refused before tar is called
    inner = tmp_path / "inner"
    (inner / "rescue").mkdir(parents=True)
    fails = pod_stop.save_results([str(inner)], str(inner / "rescue"), seen.append)
    assert fails == [str(inner)] and any("inside the directory being archived" in m for m in seen), (fails, seen)
    # tar failing or missing is an ordinary reported failure, never a crash
    orig = pod_stop.subprocess.run
    try:
        pod_stop.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 2, "", "No space left on device")
        fails = pod_stop.save_results([str(ok)], str(dest), seen.append)
    finally:
        pod_stop.subprocess.run = orig
    assert fails == [str(ok)] and any("No space left" in m for m in seen), (fails, seen)

    def gone(*a, **k):
        raise FileNotFoundError("tar")
    pod_stop.subprocess.run = gone
    try:
        fails = pod_stop.save_results([str(ok)], str(dest), seen.append)
    finally:
        pod_stop.subprocess.run = orig
    assert fails == [str(ok)] and any("save FAILED" in m for m in seen), (fails, seen)
    assert not list(dest.glob("*.part")), list(dest.iterdir())                 # no half-written archive is left behind


def test_dry_run_refuses_the_same_things_but_never_stops(tmp_path):
    """A rehearsal must prove what a real arming would do.  Every refusal still fires, the archive is still written, the stop is withheld."""
    bindir, log = _fake_runpodctl(tmp_path)
    results = _results(tmp_path)
    env = {"PATH": bindir + os.pathsep + os.environ["PATH"], "FAKE_LOG": log}
    rc, secs, out = _run(_args(tmp_path, cap_min=0.06, extra=("--save", str(results), "--dry-run")), env, 30)
    assert "DRY RUN: runpodctl will not be called" in out and "DRY RUN: this is where" in out, out
    assert (tmp_path / "volume" / "rescue" / "cedar.tgz").exists()
    assert _calls(log) == [], _calls(log)                                                # never invoked
    stale = tmp_path / "volume" / "DONE"
    rc, secs, out = _run(_args(tmp_path, extra=("--done-file", str(stale), "--dry-run")), env, 30)
    assert rc == 2 and "REFUSING" in out, (rc, out)                                      # refusals are not skipped in a dry run


# ------------------------------------------------------------------ the stop call itself

def test_only_the_pod_id_it_was_given_reaches_runpodctl():
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    seen = []
    assert pod_stop.stop_pod("pod-eeee5555", seen.append, runner=runner)
    assert calls == [["runpodctl", "stop", "pod", "pod-eeee5555"]], calls
    assert any("exit 0" in m for m in seen)


def test_stop_reports_failure_and_missing_binary():
    seen = []
    assert pod_stop.stop_pod("pod-ffff6666", seen.append, runner=lambda argv, **kw: subprocess.CompletedProcess(argv, 1, "", "boom")) is False
    assert any("exit 1" in m for m in seen), seen

    def missing(argv, **kw):
        raise FileNotFoundError(argv)

    seen = []
    assert pod_stop.stop_pod("pod-ffff6666", seen.append, runner=missing) is False
    assert any("not on the PATH" in m for m in seen), seen
