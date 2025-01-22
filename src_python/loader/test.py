path = "../data/de/"

def get_bible_vecs(file_name):
    print("Loading data from "+file_name)
    lines = []
    import codecs
    with codecs.open(path + file_name, encoding='utf-8') as f:
        for line in f:  # TODO skip first line
            if not (line.startswith("--")):
                lines.append(line.split(' ', 1)[1])
        # lines = f.readlines()

    print(lines)
    print(len(lines))
    # for line in f:
    #    print(line)
    f.close()

    from FlagEmbedding import FlagModel

    # get the BGE embedding model
    model = FlagModel('BAAI/bge-base-en-v1.5',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)

    # get the embedding of the query and corpus
    corpus_embeddings = model.encode(lines)

    print("shape of the corpus embeddings:", corpus_embeddings.shape)

    f = open(file_name, "w")

    for vec in corpus_embeddings :
        for val in vec :
            f.write(str(val))
            f.write(" ")
        f.write("\n")
    f.close()

files = ["elberfelder.txt", "luther.txt", "ne.txt", "schlachter.txt", "volxbibel.txt"]
for file in files:
    get_bible_vecs(file)

#sim_scores = query_embedding @ corpus_embeddings.T
#print(sim_scores)

#sorted_indices = sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True)
#print(sorted_indices)

#for i in sorted_indices:
#    print(f"Score of {sim_scores[i]:.3f}: \"{lines[i]}\"")


#f = open(path+"elberfelder.txt", "r")
#print(f.read())