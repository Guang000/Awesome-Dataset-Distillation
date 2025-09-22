import json

def main():
    with open("./data/update.txt", "r", encoding="utf-8") as f:
        records = [line.strip() for line in f]
    for record in records:
        location = record.split('-*-')[0]
        record = record.split('-*-')[1].replace(')', '')
        if not record.startswith('['):
            print(f"{location}\n{record}\n")
            continue
        couples = record.split('[')
        parts0 = couples[1].split('(')
        dict_record = {"title": parts0[0][:-1], "author": parts0[2].strip(), "github": None,
                       "url": parts0[1].strip(), "cite": None, "website": None}
        for couple in couples[2:]:
            parts = couple.split('(')
            if "octocat" in parts[0]:
                dict_record["github"]= parts[1]
                continue
            if "globe" in parts[0]:
                dict_record["website"] = parts[1]
                continue
            if "book" in parts[0]:
                dict_record["cite"] = parts[1].split('/')[-1].split('.')[0]
                continue
        print(location)
        print("{")
        print(json.dumps(dict_record).replace(", \"", ", \n\"")[1:-1])
        print("}\n")

if __name__ == "__main__":
    main()