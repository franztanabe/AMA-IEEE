import json, time, hashlib, platform, subprocess
from pathlib import Path
from pipeline.phase1 import run_phase1
import run_phase2, run_phase3, run_phase4

def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

def main(cfg):
    t0 = time.time()
    run_dir = Path(cfg["runs_dir"]) / f"run_{int(t0)}"
    run_dir.mkdir(parents=True)
    canonical = run_phase1.execute(cfg["ingest"], out=run_dir / "phase1")
    cfg_p2 = {**cfg["phase2"], "in_dir": canonical, "out_dir": run_dir / "phase2", "queue_dir": run_dir / "queue_p3"}
    run_phase2.main(cfg_p2)
    cfg_p3 = {**cfg["phase3"], "queue_dir": run_dir / "queue_p3", "frames_dir": run_dir / "phase2", "reid_queue": run_dir / "queue_p4", "out_dir": run_dir / "phase3"}
    run_phase3.main(cfg_p3)
    cfg_p4 = {**cfg["phase4"], "reid_queue": run_dir / "queue_p4", "frames_dir": run_dir / "phase2", "out_dir": run_dir / "phase4"}
    run_phase4.main(cfg_p4)
    meta = {
        "run_id": run_dir.name,
        "started_at": t0,
        "ended_at": time.time(),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "seed": cfg["seed"],
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return run_dir

if __name__ == "__main__":
    cfg = json.load(open("config/end2end.json"))
    main(cfg)