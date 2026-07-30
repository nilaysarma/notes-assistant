from langchain_text_splitters import RecursiveCharacterTextSplitter

text = (
    "Machine Learning is fascinating. "
    "Linear Regression is simple. "
    "Decision Trees are powerful. "
    "Neural Networks are flexible."
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10,
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}")
    print(repr(chunk))
    print("-" * 40)

search_from = 0

for i, chunk in enumerate(chunks):
    start = text.find(chunk, search_from)
    end = start + len(chunk)

    print(
        f"Chunk {i}: "
        f"start={start}, "
        f"end={end}"
    )

    search_from = start + 1