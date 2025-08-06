import pandas as pd
import json
from datetime import datetime


def main():
    # obtain all articles
    with open("./data/articles.json", "r", encoding="utf-8") as f:
        data_all = json.load(f)
    df_all = None
    for key1 in data_all.keys():
        for arr_articles in data_all[key1].values():
            df_section = pd.DataFrame(arr_articles)
            df_all = pd.concat([df_all, df_section], ignore_index=True) if df_all is not None else df_section
    # fetch latest articles
    list_latest = pd.read_json("./data/latest_names.jsonl", lines=True)
    titles = list_latest["title"]
    data_latest = df_all.loc[df_all["cite"].isin(titles)]

    # change df to dict, adding date
    # e.g. "2000/01/01": [dict_article1, dict_article2]
    dict_latest = {}
    for date in list(dict.fromkeys(list_latest["date"])):
        names = list_latest.loc[list_latest["date"] == date]["title"]
        date = datetime.strptime(str(date), "%Y%m%d").strftime("%Y/%m/%d")
        articles = []
        for name in names:
            df = data_latest.loc[data_latest["cite"] == name]
            articles.append(df.iloc[0].to_dict())
        dict_latest[date]=articles
#     print(dict_latest)
    # save to file
    with open("data/latest.json", "w", encoding="utf-8") as f:
        json.dump(dict_latest, f, ensure_ascii=False, indent=2)
    # data_latest.to_json("data/test.json", orient="records", force_ascii="False", indent=2)

if __name__ == "__main__":
    main()