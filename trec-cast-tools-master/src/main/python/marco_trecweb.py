from tqdm import tqdm
import sys
import os
import io
import codecs
from trecweb_utils import convert_to_trecweb, add_passage_ids
from passage_chunker import SpacyPassageChunker

def parse_sim_file(filename):
    """Reads the deduplicated documents file and stores the 
    duplicate passage ids into a dictionary

    Args:
        filename (str): Path to MARCO duplicates file

    Returns:
        dict: MARCO duplicates lookup dictionary
    """

    sim_dict = {}
    lines = open(filename).readlines()
    for line in lines:
        data = line.strip().split(':')
        if len(data[1]) > 0:
            sim_docs = data[-1].split(',')
            for docs in sim_docs:
                sim_dict[docs] = 1
                
    print("There are {} duplicates".format(len(sim_dict)))
    return sim_dict


def write_document(line, fp, sim_dict, passageChunker):
    """Writes MARCO doc to trecweb

    Args:
        line : Marco doc
        fp : trecweb file path
        sim_dict : duplicates lookup dict
    """
    try:
        idx, url, title, body = line.strip().split('\t')
        
        # if the id is a duplicate, don't add it
        if idx in sim_dict:
            return
        
        
        idx = 'MARCO_' + str(idx)

        passageChunker.sentence_tokenization(body)
        

        passages = passageChunker.create_passages()
        
        passage_splits = add_passage_ids(passages)

        trecweb_format = convert_to_trecweb(idx, title, passage_splits, url)
        fp.write(trecweb_format)

    except:
        #either idx, url, title, or body is missing
        return

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("USAGE: python3 marco_trecweb.py path_to_collection.tsv path_of_dumpdir duplicates_file")
        exit(0)
    
    marco_file = sys.argv[1]
    dump_dir = sys.argv[2]
    sim_file = sys.argv[3]

    # Create the directory (for dumping files) if it doesn't exists
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    
    print("Loading similarity file.")
    sim_dict = parse_sim_file(sim_file)

    input_file = os.path.basename(marco_file)

    print("Starting processing.")
    print("Output directory: " + dump_dir)
    dumper_file = os.path.join(dump_dir, input_file + '.trecweb')
    print("Writing output to: " + dumper_file)
    fp = codecs.open(dumper_file, 'w', 'utf-8')

    passageChunker = SpacyPassageChunker()

    with io.open(marco_file, "r", encoding="utf-8") as input:

        for line in tqdm(input, total=3213835):
            write_document(line, fp, sim_dict, passageChunker)

    fp.close()
