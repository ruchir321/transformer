import pandas

with open("data/wikihow.txt", "w", encoding="utf-8") as f:
    for chunk in pandas.read_csv("data/wikihowAll.csv", chunksize=1024):
        for index, row in chunk.iterrows():
            f.write(f"headline: {row['headline']}\n")
            f.write(f"title: {row['title']}\n")
            f.write(f"text: {row['text']}\n")
            f.write("---\n\n")  # Separator between rows
