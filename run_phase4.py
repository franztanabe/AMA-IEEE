import json, numpy as np, csv
from pathlib import Path
from reid.face import PetFaceEmbedder
from reid.body import BodyPartEmbedder
from reid.catalog import IndividualCatalog
from reid.fusion import late_fusion
from pipeline.io import open_image, crop_face, crop_body

def main(cfg):
    face = PetFaceEmbedder(cfg["petface_weights"])
    body = BodyPartEmbedder(cfg["wildlife_weights"])
    cat = IndividualCatalog.load(cfg["catalog_dir"])
    tau = cfg["openset_threshold"]
    out_csv = Path(cfg["out_dir"]) / "phase4_reid.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "bbox", "match_id", "rank1_dist", "rank5_ids", "is_novel"])
        for q in sorted(Path(cfg["reid_queue"]).glob("*.json")):
            p = json.load(q.open())
            img = open_image(Path(cfg["frames_dir"]) / f"{p['frame_id']}.jpg")
            roi_face = crop_face(img, p["bbox"])
            roi_body = crop_body(img, p["bbox"])
            e_face = face.embed(roi_face)
            e_body = body.embed(roi_body)
            e_fused = late_fusion(e_face, e_body, w_face=cfg["w_face"])
            ranked = cat.rank(e_fused, k=5)
            is_novel = ranked[0].dist > tau
            w.writerow([p["frame_id"], p["bbox"], None if is_novel else ranked[0].id, ranked[0].dist, [r.id for r in ranked], is_novel])
            if is_novel:
                cat.stage_for_review(e_fused, p)
    return out_csv

if __name__ == "__main__":
    cfg = json.load(open("config/phase4.json"))
    main(cfg)