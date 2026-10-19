def scanning_process(list_of_dirs, list_of_ignored_dirs, model_path=None):
    from safenet import SafeDetector, all_labels_sfw
    from liteindex import DefinedIndex
    from glob import iglob
    import json
    import os
    import hashlib

    images_index = DefinedIndex(
        "images_index",
        schema={
            "image_path": "string",
            "image_size": "number",
            "image_hash": "string",
            "detections": "string",
            **{label: "number" for label in all_labels_sfw},
        },
        db_path="images_index.db",
    )

    detector = SafeDetector(model_path=model_path)

    for dir in list_of_dirs:
        for file in iglob(os.path.join(dir, "**", "*.*"), recursive=True):
            if os.path.isfile(file):
                if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png"]:
                    if not any(
                        ignored_dir in file for ignored_dir in list_of_ignored_dirs
                    ):
                        preds = detector.detect(file)
                        data = {
                            prediction["class"]: float(prediction["score"])
                            for prediction in preds
                            if prediction["class"] in all_labels_sfw
                        }
                        data["detections"] = json.dumps(preds)
                        data["image_path"] = file
                        data["image_size"] = os.path.getsize(file)
                        data["image_hash"] = hashlib.sha256(
                            open(file, "rb").read()
                        ).hexdigest()

                        images_index.update({data["image_hash"]: data})


if __name__ == "__main__":
    scanning_process(["/Users/praneeth.bedapudi/Desktop"], [])
