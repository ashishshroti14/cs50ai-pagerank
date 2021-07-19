import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probab_dict  = {}
    def find_probab(some_page):
        
        corpus_page_probab_1 = 0.0
        corpus_page_probab_2 = 0.0

        if corpus[page] == 0:
            return 1/len(corpus)
        if some_page  in corpus[page]:
            
            corpus_page_probab_1 = 1/len(corpus[page])
    
        corpus_page_probab_2 = 1/len(corpus)
        
        return (1-damping_factor)*corpus_page_probab_2 + (damping_factor) * corpus_page_probab_1

    for corpus_page in corpus:
        probab_dict[corpus_page] = find_probab(corpus_page)

    return probab_dict
    # raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    pagerank_dict = dict()
    counts = dict()
    for i in range(n):
        
        if i == 0:
            sample = random.choice(list(corpus))

        model = transition_model(corpus, sample, damping_factor)
        collection = []
        list(map((lambda page : collection.extend( [page]*(int(model[page] * n)))),  model))

        sample = random.choice(list(collection))


        for corpus_page in corpus:
            
            if sample == corpus_page:
                if corpus_page not in counts:
                    counts.update({corpus_page: 0})
                counts[corpus_page] += 1
                
    pagerank_dict = {k: v/n for k, v in counts.items()}
    return pagerank_dict

    


        

    # raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank_dict = dict()
    num = len(corpus)
    for page in corpus: 
        pagerank_dict.update({page: num})
    
    
    while True:
        is_final_count = 0

        temp = copy.deepcopy(pagerank_dict)
        for page1 in corpus:
            probab_sum = 0
            for page2 in corpus:
                if corpus[page2] == 0:
                    probab_sum += 1/len(corpus)
                    continue
                if page1 in corpus[page2]:
                    probab_sum += pagerank_dict[page2]/len(corpus[page2])
            pagerank_dict[page1] = (1-damping_factor)/len(corpus) + damping_factor * probab_sum 
        
        
        for page in pagerank_dict:
            if (temp[page] - pagerank_dict[page] < 0.001 and temp[page] - pagerank_dict[page] > -0.001 ) :
                is_final_count +=1

        if is_final_count == len(corpus):
            break
        

    return pagerank_dict
    # raise NotImplementedError


if __name__ == "__main__":
    main()
