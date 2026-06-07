import json, csv
from pathlib import Path
from speciesnet import SpeciesNetClassifier
from pipeline.io import open_image, crop
from pipeline.taxonomy import is_felis_catus

def main(cfg):
    clf = SpeciesNetClassifier(model_id=cfg["speciesnet_release"], taxonomy=cfg["taxonomy_csv"])
    out_csv = Path(cfg["out_dir"]) / "phase3_classifications.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "bbox", "top1_class", "top1_score", "top5_classes", "is_felis_catus"])
        for q in sorted(Path(cfg["queue_dir"]).glob("*.json")):
            payload = json.load(q.open())
            img = open_image(Path(cfg["frames_dir"]) / f"{payload['frame_id']}.jpg")
            for det in payload["dets"]:
                roi = crop(img, det["bbox"])
                pred = clf.classify(roi, topk=5)
                w.writerow([payload["frame_id"], det["bbox"], pred.top1.label, pred.top1.score, [p.label for p in pred.topk], is_felis_catus(pred.top1)])
                if is_felis_catus(pred.top1):
                    json.dump({"frame_id": payload["frame_id"], "bbox": det["bbox"], "label": pred.top1.label, "score": pred.top1.score}, (Path(cfg["reid_queue"]) / f"{payload['frame_id']}_{det['bbox'][0]}.json").open("w"))
    return out_csv

if __name__ == "__main__":
    cfg = json.load(open("config/phase3.json"))
    main(cfg)