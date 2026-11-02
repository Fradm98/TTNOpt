#!/usr/bin/env python3
"""Live monitor for your PBS jobs (qsub/qstat).

For every job of the user it shows: state, node, cores (requested and
effectively used from cpupercent), memory used vs requested, walltime used
vs requested, CPU time, start time and elapsed/queued time. Refreshes in
place like `top`; Ctrl-C to quit.

Usage (on the cluster login node):
    python3 qwatch.py                 # refresh every 10 s
    python3 qwatch.py -n 30           # refresh every 30 s
    python3 qwatch.py -x              # also show recently finished jobs
    python3 qwatch.py --once          # print once and exit
    python3 qwatch.py -u otheruser
    python3 qwatch.py -cl             # whole cluster: totals, per user, per node
    python3 qwatch.py -cl -u someone  # one user's share of the cluster

Only stdlib, Python >= 3.6. Parses the plain `qstat -f` text format so it
works on PBS Pro / OpenPBS / Torque without `-F json`.
"""
import argparse
import getpass
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

STATE_NAMES = {
    "R": "RUN", "Q": "QUEUE", "H": "HOLD", "E": "EXIT", "F": "DONE",
    "X": "DONE", "B": "ARRAY", "S": "SUSP", "W": "WAIT", "T": "MOVE",
    "C": "DONE", "M": "MOVED", "U": "SUSP",
}
STATE_COLOR = {"R": "32", "Q": "33", "H": "35", "E": "36", "F": "90", "X": "90", "C": "90"}

_UNIT = {"b": 1, "w": 8, "kb": 1024, "kw": 8 * 1024, "mb": 1024 ** 2, "mw": 8 * 1024 ** 2,
         "gb": 1024 ** 3, "gw": 8 * 1024 ** 3, "tb": 1024 ** 4, "tw": 8 * 1024 ** 4}


def parse_qstat_f(text):
    """Return a list of dicts, one per job, from `qstat -f` output."""
    jobs, cur, last_key = [], None, None
    for line in text.splitlines():
        if line.startswith("Job Id:"):
            cur = {"Job Id": line.split(":", 1)[1].strip()}
            jobs.append(cur)
            last_key = None
        elif cur is None or not line.strip():
            continue
        elif line.startswith("\t") and last_key is not None:
            # continuation of a wrapped long attribute value
            cur[last_key] += line.strip()
        elif " = " in line:
            key, val = line.strip().split(" = ", 1)
            cur[key] = val
            last_key = key
    return jobs


def mem_bytes(s):
    if not s:
        return None
    m = re.match(r"^\s*([\d.]+)\s*([a-zA-Z]*)\s*$", s)
    if not m:
        return None
    return float(m.group(1)) * _UNIT.get(m.group(2).lower() or "b", 1)


def fmt_mem(b):
    if b is None:
        return "-"
    for unit, div in (("T", 1024 ** 4), ("G", 1024 ** 3), ("M", 1024 ** 2), ("K", 1024)):
        if b >= div:
            v = b / div
            return ("%.1f%s" % (v, unit)) if v < 100 else ("%d%s" % (v, unit))
    return "%dB" % b


def hms_seconds(s):
    """'HH:MM:SS' (hours may exceed 24) or plain seconds -> int seconds."""
    if not s:
        return None
    parts = s.strip().split(":")
    try:
        parts = [int(float(p)) for p in parts]
    except ValueError:
        return None
    sec = 0
    for p in parts:
        sec = sec * 60 + p
    return sec


def fmt_dur(sec):
    if sec is None:
        return "-"
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return "%dd%02dh%02dm" % (d, h, m)
    return "%d:%02d:%02d" % (h, m, s)


def parse_time(s):
    """PBS timestamps look like 'Tue Oct 28 10:12:01 2026'."""
    if not s:
        return None
    try:
        return datetime.strptime(" ".join(s.split()), "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None


def nodes_of(job):
    """Unique host names from exec_host ('n01/0*14+n02/0*14') or exec_vnode."""
    raw = job.get("exec_host") or job.get("exec_vnode") or ""
    hosts = []
    for chunk in raw.replace("(", "").replace(")", "").split("+"):
        h = chunk.split("/")[0].split(":")[0].strip()
        if h and h not in hosts:
            hosts.append(h)
    return ",".join(hosts) if hosts else "-"


def parse_pbsnodes(text):
    """Return a list of dicts, one per node, from `pbsnodes -a` output."""
    nodes, cur = [], None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            cur = {"name": line.strip()}
            nodes.append(cur)
        elif cur is not None and " = " in line:
            key, val = line.strip().split(" = ", 1)
            cur[key] = val
    return nodes


def _int(s):
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def cluster_usage(nodes):
    """Totals over usable nodes; offline/down nodes are only counted apart."""
    usage = {"nodes": 0, "unusable": 0, "cores": 0, "cores_used": 0, "mem": 0, "mem_used": 0}
    for n in nodes:
        if any(s in n.get("state", "") for s in ("down", "offline", "unknown", "stale")):
            usage["unusable"] += 1
            continue
        usage["nodes"] += 1
        usage["cores"] += _int(n.get("resources_available.ncpus") or n.get("np") or n.get("pcpus"))
        usage["cores_used"] += _int(n.get("resources_assigned.ncpus"))
        usage["mem"] += mem_bytes(n.get("resources_available.mem")) or 0
        usage["mem_used"] += mem_bytes(n.get("resources_assigned.mem")) or 0
    return usage


def owner_of(job):
    return job.get("Job_Owner", "").split("@")[0]


def vnode_chunks(job):
    """(node, cores, mem bytes) per chunk of exec_vnode '(n01:ncpus=14:mem=1kb)+(n02:...)'."""
    raw = job.get("exec_vnode", "").replace("(", "").replace(")", "")
    for chunk in raw.split("+"):
        parts = chunk.strip().split(":")
        if parts[0]:
            attrs = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            yield parts[0], _int(attrs.get("ncpus")), mem_bytes(attrs.get("mem")) or 0


def render_users(all_jobs, usage):
    """One row per user: running/queued jobs and the cores/memory they hold."""
    users = {}
    for j in all_jobs:
        u = users.setdefault(owner_of(j), {"run": 0, "queued": 0, "cores": 0, "mem": 0})
        if j.get("job_state") == "R":
            u["run"] += 1
            u["cores"] += _int(j.get("Resource_List.ncpus"))
            u["mem"] += mem_bytes(j.get("Resource_List.mem")) or 0
        elif j.get("job_state") == "Q":
            u["queued"] += 1

    def row(name, run, queued, cores, mem):
        cores_pct = 100.0 * cores / usage["cores"] if usage["cores"] else 0
        mem_pct = 100.0 * mem / usage["mem"] if usage["mem"] else 0
        return "%-14s %4s %4s %6d %5.1f%% %8s %5.1f%%" % (
            name[:14], run, queued, cores, cores_pct, fmt_mem(mem), mem_pct)

    out = ["%-14s %4s %4s %6s %6s %8s %6s" % ("USER", "RUN", "QUE", "CORES", "%", "MEM", "%")]
    order = sorted(users.items(), key=lambda kv: (-kv[1]["mem"], -kv[1]["cores"], kv[0]))
    for name, u in order:
        out.append(row(name, u["run"], u["queued"], u["cores"], u["mem"]))
    # Assigned on the nodes but not matched to any visible job: other users'
    # jobs are hidden from qstat on clusters that disable query_other_jobs.
    hidden_cores = usage["cores_used"] - sum(u["cores"] for u in users.values())
    hidden_mem = usage["mem_used"] - sum(u["mem"] for u in users.values())
    if hidden_cores > 0 or hidden_mem > 0:
        out.append(row("(not visible)", "-", "-", max(hidden_cores, 0), max(hidden_mem, 0)))
    out.append(row("(free)", "-", "-", max(usage["cores"] - usage["cores_used"], 0),
                   max(usage["mem"] - usage["mem_used"], 0)))
    return out


def render_nodes(all_jobs, nodes, only_user=None):
    """One row per node, so a job that fits no single node is visible.
    With only_user: just the nodes that user is on, listing only that user."""
    held = {}
    for j in all_jobs:
        if j.get("job_state") == "R":
            if only_user and owner_of(j) != only_user:
                continue
            for node, cores, mem in vnode_chunks(j):
                cur = held.setdefault(node, {}).setdefault(owner_of(j), [0, 0])
                cur[0] += cores
                cur[1] += mem
    out = ["%-10s %-10s %9s %16s %8s  %s" % ("NODE", "STATE", "CORES", "MEM", "FREE MEM", "USERS")]
    for n in nodes:
        name = n["name"]
        if only_user and name not in held:
            continue
        total_cores = _int(n.get("resources_available.ncpus") or n.get("np") or n.get("pcpus"))
        total_mem = mem_bytes(n.get("resources_available.mem")) or 0
        used_mem = mem_bytes(n.get("resources_assigned.mem")) or 0
        who = sorted(held.get(name, {}).items(), key=lambda kv: -kv[1][1])
        out.append("%-10s %-10s %9s %16s %8s  %s" % (
            name[:10], n.get("state", "?")[:10],
            "%d/%d" % (_int(n.get("resources_assigned.ncpus")), total_cores),
            "%s/%s" % (fmt_mem(used_mem), fmt_mem(total_mem)),
            fmt_mem(max(total_mem - used_mem, 0)),
            ", ".join("%s %dc/%s" % (u, c, fmt_mem(m)) for u, (c, m) in who)))
    return out


def render_cluster(all_jobs, user, nodes, use_color):
    """Cluster view: totals plus per-user/per-node tables, or one user's share."""
    usage = cluster_usage(nodes)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if user:
        mine = [j for j in all_jobs if owner_of(j) == user]
        running = [j for j in mine if j.get("job_state") == "R"]
        queued = [j for j in mine if j.get("job_state") == "Q"]
        title = "Cluster usage of %s  \u2014  %s  \u2014  %d running, %d queued" % (
            user, now, len(running), len(queued))
        cores = sum(_int(j.get("Resource_List.ncpus")) for j in running)
        mem = sum(mem_bytes(j.get("Resource_List.mem")) or 0 for j in running)
    else:
        title = "Cluster usage  \u2014  %s" % now
        cores, mem = usage["cores_used"], usage["mem_used"]

    def line(label, held, used, total, fmt):
        if not total:
            return "%-6s total not reported by pbsnodes" % label
        pct = 100.0 * held / total
        text = "%-6s %s %s / %s (%.1f%%)" % (label, bar(pct, 20, use_color), fmt(held), fmt(total), pct)
        if user:
            text += "   all users %s (%.0f%%)" % (fmt(used), 100.0 * used / total)
        return text + "   free %s" % fmt(max(total - used, 0))

    out = [
        color("1", title, use_color),
        "%d usable nodes, %d down/offline" % (usage["nodes"], usage["unusable"]),
        line("CORES", cores, usage["cores_used"], usage["cores"], str),
        line("MEM", mem, usage["mem_used"], usage["mem"], fmt_mem),
        "",
    ]
    if user:
        out += render_nodes(all_jobs, nodes, only_user=user) if running else ["  (no running jobs)"]
    else:
        out += render_users(all_jobs, usage) + [""] + render_nodes(all_jobs, nodes)
    out += ["", color("90", "Cores and memory are what jobs reserved, not what they use. Ctrl-C to quit.", use_color)]
    return "\n".join(out)


def color(code, s, enabled):
    return "\033[%sm%s\033[0m" % (code, s) if enabled and code else s


def pct_color(p):
    if p is None:
        return None
    return "31" if p >= 90 else ("33" if p >= 75 else "32")


def bar(p, width, enabled):
    if p is None:
        return " " * (width + 2)
    filled = int(round(min(p, 100) / 100.0 * width))
    return "[" + color(pct_color(p), "#" * filled, enabled) + "." * (width - filled) + "]"


def build_row(job, now):
    st = job.get("job_state", "?")
    mem_used = mem_bytes(job.get("resources_used.mem"))
    mem_req = mem_bytes(job.get("Resource_List.mem"))
    wall_used = hms_seconds(job.get("resources_used.walltime"))
    wall_req = hms_seconds(job.get("Resource_List.walltime"))
    cput = hms_seconds(job.get("resources_used.cput"))
    ncpus = job.get("Resource_List.ncpus") or job.get("resources_used.ncpus") or "-"
    cpupct = job.get("resources_used.cpupercent")
    eff = ("%.1f" % (float(cpupct) / 100.0)) if cpupct not in (None, "") else "-"

    stime = parse_time(job.get("stime"))
    qtime = parse_time(job.get("qtime"))
    if st == "R" and stime:
        started = stime.strftime("%d/%m %H:%M")
        elapsed = fmt_dur((now - stime).total_seconds())
    elif st in ("Q", "H", "W") and qtime:
        started = "queued"
        elapsed = "wait " + fmt_dur((now - qtime).total_seconds())
    else:
        started = stime.strftime("%d/%m %H:%M") if stime else "-"
        elapsed = fmt_dur(wall_used)

    mem_pct = 100.0 * mem_used / mem_req if mem_used is not None and mem_req else None
    wall_pct = 100.0 * wall_used / wall_req if wall_used is not None and wall_req else None
    return {
        "id": job["Job Id"].split(".")[0],
        "name": job.get("Job_Name", "-"),
        "state": st,
        "queue": job.get("queue", "-"),
        "node": nodes_of(job),
        "cores": "%s/%s" % (eff, ncpus),
        "mem": "%s/%s" % (fmt_mem(mem_used), fmt_mem(mem_req)),
        "mem_pct": mem_pct,
        "wall": "%s/%s" % (fmt_dur(wall_used), fmt_dur(wall_req)),
        "wall_pct": wall_pct,
        "cput": fmt_dur(cput),
        "started": started,
        "elapsed": elapsed,
        "comment": job.get("comment", ""),
    }


def render(jobs, user, use_color, show_comment):
    now = datetime.now()
    rows = [build_row(j, now) for j in jobs]
    width = shutil.get_terminal_size((160, 40)).columns
    name_w = max([4] + [len(r["name"]) for r in rows])
    name_w = min(name_w, 28)

    header = ("%-9s %-*s %-5s %-10s %-9s %-15s %-19s %-24s %-9s %-11s %s" % (
        "JOBID", name_w, "NAME", "STATE", "QUEUE", "CORES", "NODE",
        "MEM used/req", "WALL used/limit", "CPUTIME", "START", "ELAPSED"))
    out = []
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    summary = "  ".join("%s:%d" % (STATE_NAMES.get(k, k), v) for k, v in sorted(counts.items()))
    out.append(color("1", "PBS jobs of %s  —  %s  —  %d jobs  %s" % (
        user, now.strftime("%Y-%m-%d %H:%M:%S"), len(rows), summary), use_color))
    out.append(color("1;4", header[:width], use_color))
    for r in rows:
        st = r["state"]
        mem_pct = ("%3d%%" % r["mem_pct"]) if r["mem_pct"] is not None else "    "
        wall_pct = ("%3d%%" % r["wall_pct"]) if r["wall_pct"] is not None else "    "
        mem_cell = "%-13s %s" % (r["mem"], color(pct_color(r["mem_pct"]), mem_pct, use_color))
        wall_cell = "%-18s %s" % (r["wall"], color(pct_color(r["wall_pct"]), wall_pct, use_color))
        line = "%-9s %-*s %s %-10s %-9s %-15s %s  %s  %-9s %-11s %s" % (
            r["id"], name_w, r["name"][:name_w],
            color(STATE_COLOR.get(st), "%-5s" % STATE_NAMES.get(st, st), use_color),
            r["queue"][:10], r["cores"], r["node"][:15],
            mem_cell, wall_cell, r["cput"], r["started"], r["elapsed"])
        out.append(line)
        if st == "R":
            out.append(" " * (9 + 1 + name_w + 1 + 6 + 11 + 10 + 16)
                       + "mem " + bar(r["mem_pct"], 12, use_color)
                       + "      wall " + bar(r["wall_pct"], 12, use_color))
        if show_comment and r["comment"]:
            out.append(color("90", "          └ " + r["comment"][:width - 12], use_color))
    if not rows:
        out.append("  (no jobs)")
    out.append("")
    out.append(color("90", "CORES = effective cores in use (cpupercent/100) / requested ncpus. "
                           "Ctrl-C to quit.", use_color))
    return "\n".join(out)


def fetch(history, from_file):
    if from_file:
        with open(from_file) as fh:
            text = fh.read()
    else:
        cmd = ["qstat", "-f"]
        if history:
            cmd.append("-x")  # PBS Pro: include finished jobs still in history
        try:
            text = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired) as e:
            return None, "qstat failed: %s" % e
    return parse_qstat_f(text), None


def fetch_nodes(from_file):
    """Parsed `pbsnodes -a`, or None if pbsnodes is unavailable on this machine."""
    if from_file:
        with open(from_file) as fh:
            text = fh.read()
    else:
        try:
            text = subprocess.run(["pbsnodes", "-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
    return parse_pbsnodes(text) or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-u", "--user", default=None,
                    help="job owner to show ('*' for everyone); default: you, or with -cl the whole cluster")
    ap.add_argument("-cl", "--cluster", action="store_true",
                    help="show usage of the whole cluster, or of one user's share with -u")
    ap.add_argument("-n", "--interval", type=float, default=10.0, help="refresh seconds (default 10)")
    ap.add_argument("-x", "--history", action="store_true", help="include finished jobs (qstat -x)")
    ap.add_argument("-c", "--comments", action="store_true", help="show the scheduler comment per job")
    ap.add_argument("--once", action="store_true", help="print once and exit")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--from-file", help=argparse.SUPPRESS)  # debugging: parse a saved qstat -f dump
    ap.add_argument("--nodes-from-file", help=argparse.SUPPRESS)  # debugging: saved pbsnodes -a dump
    args = ap.parse_args()
    use_color = sys.stdout.isatty() and not args.no_color
    user = args.user or os.environ.get("USER") or getpass.getuser()
    cluster_user = None if args.user == "*" else args.user

    try:
        while True:
            all_jobs, err = fetch(args.history, args.from_file)
            if err:
                screen = err
            elif args.cluster:
                nodes = fetch_nodes(args.nodes_from_file)
                screen = (render_cluster(all_jobs, cluster_user, nodes, use_color) if nodes
                          else "pbsnodes returned nothing: cluster usage unavailable")
            else:
                jobs = [j for j in all_jobs if user == "*" or owner_of(j) == user]
                screen = render(jobs, user, use_color, args.comments)
            if args.once:
                print(screen)
                return
            sys.stdout.write("\033[H\033[2J" + screen + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
