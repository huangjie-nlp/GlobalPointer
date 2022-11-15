import json


def processing_clue_data(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            d = json.loads(line.strip("\n"))
            sample = {"text": d["text"]}
            labels = d["label"]
            ners = []
            for k, v in labels.items():
                ner_type = k
                for i, j in v.items():
                    ner_name = i
                    for x in j:
                        ner = str(x[0]) + '$' +str(x[1]) + '$' +ner_type + '@' + ner_name
                        ners.append(ner)
            sample["ners"] = ners
            data.append(sample)
    return data

def generate_schema(fn):
    data = json.load(open(fn, "r", encoding="utf-8"))
    id2label = {}
    for k, v in data.items():
        id2label[v] = k
    return [data, id2label]

if __name__ == '__main__':
    file = "../dataset/cluener/dev.json"
    data = processing_clue_data(file)
    json.dump(data, open("../dataset/cluener/dev_data.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    # file = "../dataset/cluener/ent2id.json"
    # schema = generate_schema(file)
    # json.dump(schema, open("../dataset/cluener/schema.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)