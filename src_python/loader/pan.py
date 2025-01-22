from sympy.solvers.diophantine.diophantine import length

path = "../data/pan11/01-manual-obfuscation-highjac/"
PAN11_PREFIX_SUSP = path+"susp/suspicious-document";
PAN11_PREFIX_SRC  = path+"src/source-document";

prefix_sentence_susp = "../data/pan11/sentences/susp/"
prefix_sentence_src = "../data/pan11/sentences/src/"

plagiarism_pairs=[
                 ["00228", "05889"]
                ,["00574", "06991"]
                ,["00574", "06586"]
                ,["00815", "01537"]
                ,["04617", "01107"]
                ,["10751", "06521"]
                ,["02161", "06392"]
                ,["02841", "10886"]
                ,["04032", "07742"]
                ,["04032", "02661"]
                ,["04032", "07640"]
                ,["04751", "08779"]
                ,["04953", "00732"]
                ,["08405", "10603"]
                ,["09029", "03302"]
                ,["09922", "10065"]
                ,["08405", "10603"]
                ,["10497", "06489"]
]

def get_text_sentence_wise(file_name : str) -> list[str]:
    print("Loading data from " + file_name)
    lines = ""
    import codecs

    with codecs.open(file_name, encoding='utf-8') as f:
        for line in f:
            lines += line
        # lines = f.readlines()
    f.close()
    # print(lines)
    lines = lines.replace("\n", " ")  # replace line breaks
    lines = lines.replace("\r", " ")  # replace line breaks
    #lines = lines.replace("\ufeff", " ")  # replace strange characters
    #lines = lines.replace("\x8d", " ")  # replace strange characters
    print(lines)
    # sentences = lines.split("")
    #import nltk
    #nltk.download('punkt_tab')
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(lines)
    return sentences

def write_file(file_name, sentences):
    print("Writing to " + file_name)
    import codecs
    with codecs.open(file_name,"w", encoding='utf-8') as f:
        for sentence in sentences :
            f.write(sentence.strip())
            f.write("\n")
    f.close()

def get_pan_vecs(pair_num):
    file_name_susp = PAN11_PREFIX_SUSP + plagiarism_pairs[pair_num][0] +".txt"
    file_name_src  = PAN11_PREFIX_SRC  + plagiarism_pairs[pair_num][1] + ".txt"

    setences_sups = get_text_sentence_wise(file_name_susp)
    setences_src  = get_text_sentence_wise(file_name_src)

    print("--------------Text suspicious pair susp " + str(pair_num))
    for sentence in setences_sups:
        print(sentence)
    print("--------------Text suspicious pair src " + str(pair_num))
    for sentence in setences_src:
        print(sentence)
    write_file(prefix_sentence_susp + plagiarism_pairs[pair_num][0] +".txt", setences_sups)
    write_file(prefix_sentence_src  + plagiarism_pairs[pair_num][1] + ".txt", setences_src)

    from FlagEmbedding import FlagModel

    # get the BGE embedding model
    model = FlagModel('BAAI/bge-base-en-v1.5',
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)

    # get the embedding of the query and corpus
    susp_embeddings = model.encode(setences_sups)
    print("shape of the corpus embeddings:", susp_embeddings.shape)
    f = open(prefix_sentence_susp + plagiarism_pairs[pair_num][0] +".vec", "w")

    for vec in susp_embeddings:
        for val in vec:
            f.write(str(val))
            f.write(" ")
        f.write("\n")
    f.close()
    # now the embedding of the src document
    src_embeddings = model.encode(setences_src)
    print("shape of the corpus embeddings:", src_embeddings.shape)
    f = open(prefix_sentence_src + plagiarism_pairs[pair_num][1] +".vec", "w")

    for vec in src_embeddings:
        for val in vec:
            f.write(str(val))
            f.write(" ")
        f.write("\n")
    f.close()

#get_pan_vecs(0)

for x in range(len(plagiarism_pairs)):
    get_pan_vecs(x)

