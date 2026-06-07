import json, hashlib, csv
from pathlib import Path
from pytorch_wildlife.models import detection as pwd
from pipeline.io import iter_canonical_frames, write_anonymized_frame
from pipeline.privacy import blur_bbox

def main(cfg):
    model = pwd.MegaDetectorV6(pretrained=True, weights_hash=cfg["weights_hash_md"])
    model.eval()
    out_csv = Path(cfg["out_dir"]) / "phase2_detections.csv"
    thr = cfg["confidence_threshold"]
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "x", "y", "w", "h", "class", "score"])
        for frame in iter_canonical_frames(cfg["in_dir"]):
            dets = model.detect(frame.tensor, conf=thr)
            keep_for_phase3 = []
            for det in dets:
                if det.cls == "person":
                    frame.image = blur_bbox(frame.image, det.bbox, sigma=cfg["blur_sigma"])
                    w.writerow([frame.id, det.bbox, "person_masked", det.score])
                elif det.cls == "vehicle":
                    w.writerow([frame.id, det.bbox, "vehicle", det.score])
                elif det.cls == "animal":
                    keep_for_phase3.append(det)
                    w.writerow([frame.id, det.bbox, "animal", det.score])
            write_anonymized_frame(cfg["out_dir"], frame)
            if keep_for_phase3:
                json.dump({"frame_id": frame.id, "dets": [d.to_dict() for d in keep_for_phase3]}, (Path(cfg["queue_dir"]) / f"{frame.id}.json").open("w"))
    return out_csv

if __name__ == "__main__":
    cfg = json.load(open("config/phase2.json"))
    main(cfg)