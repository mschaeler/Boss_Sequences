#!/usr/bin/env python
# coding=UTF-8
""" Plagiarism detection system.

    Program for the PAN 2014 Plagiarism Detection task.
    The code structure is based in the structure provided by Arnd Oberlaender as baseline

    The system is described in the following paper(s):

    Miguel Sanchez-Perez, Grigori Sidorov, Alexander Gelbukh. 
    The Winning Approach to Text Alignment for Text Reuse Detection at PAN 2014. 
    In: L. Cappellato, N. Ferro, M. Halvey, W. Kraaij (eds.). 
    Notebook for PAN at CLEF 2014. 
    CLEF 2014. CLEF2014 Working Notes. Sheffield, UK, September 15-18, 2014. 
    CEUR Workshop Proceedings, ISSN 1613-0073, Vol. 1180, CEUR-WS.org, 2014, pp. 1004�1011.
    
    (This paper can be also mistakenly indexed as �A Winning Approach to Text Alignment for Text Reuse Detection at PAN 2014�.)

    http://ceur-ws.org/Vol-1180/CLEF2014wn-Pan-SanchezPerezEt2014.pdf
    http://www.gelbukh.com/plagiarism-detection/PAN-2014/A%20Winning%20Approach%20to%20Text%20Alignment%20for%20Text%20Reuse%20Detection%20at%20PAN%202014.pdf

    License: free for non-commercial academic purposes. 
    Any publication that benefited from these data or software must state the origin of the data and software and cite the abovementioned paper(s). 
    We will be grateful to you if you let us know of the use of the data or software and of citing our papers. 
    Any derived work should specify the original source and its authors and contain this license, including the publication references mentioned above. 
    If you modify this corpus or software, correct errors in it, or add annotation/functionality to it, we will be grateful if you send us the new version, to be available from this site. 
    See also individual license files or comments in the specific files, if any.
"""
__authors__ = 'Miguel Angel Sanchez Perez, Alexander Gelbukh and Grigori Sidorov'
__email__ = 'masp1988 at hotmail dot com'
__version__ = '3.0'

import os
import sys
import xml.dom.minidom
import codecs
import nltk
import re
import math
import time
import flist

def sum_vect(dic1,dic2):
    res=dic1
    for i in dic2.keys():
        if res.has_key(i):
            res[i]+=dic2[i]
        else:
            res[i]=dic2[i]
    return res

def ss_treat(list_dic,offsets,min_sentlen,rssent):
    if rssent=='no':
        i=0
        range_i=len(list_dic)-1
        while i<range_i:
            if sum(list_dic[i].values())<min_sentlen:
                list_dic[i+1]=sum_vect(list_dic[i+1],list_dic[i])
                del list_dic[i]
                offsets[i+1]=(offsets[i][0],offsets[i+1][1]+offsets[i][1])
                del offsets[i]
                range_i-=1
            else:
                i=i+1
    else:
        i=0
        range_i=len(list_dic)-1
        while i<range_i:
            if sum(list_dic[i].values())<min_sentlen:
                del list_dic[i]
                del offsets[i]
                range_i-=1
            else:
                i=i+1

def tf_idf(list_dic1,voc1,list_dic2,voc2):
    idf=sum_vect(voc1,voc2)
    td=len(list_dic1)+len(list_dic2)
    for i in range(len(list_dic1)):
        for j in list_dic1[i].keys():
            list_dic1[i][j]*=math.log(td/float(idf[j]))
    for i in range(len(list_dic2)):
        for j in list_dic2[i].keys():
            list_dic2[i][j]*=math.log(td/float(idf[j]))
            
def eucl_norm(d1):
        norm=0.0
        for val in d1.values():
            norm+=float(val*val)
        return math.sqrt(norm)

def cosine_measure(d1,d2):
    dot_prod=0.0
    det=eucl_norm(d1)*eucl_norm(d2)
    if det==0:
        return 0 
    for word in d1.keys():
        if d2.has_key(word):
            dot_prod+=d1[word]*d2[word]
    return dot_prod/det

def dice_coeff(d1,d2):
    if len(d1)+len(d2)==0:
        return 0
    intj=0
    for i in d1.keys():
        if d2.has_key(i):
            intj+=1
    return 2*intj/float(len(d1)+len(d2))

def adjacent(a,b,th):
    if abs(a-b)-th-1<=0:
        return True
    else:
        return False

def integrate_cases(ps,src_gap,susp_gap,src_size,susp_size):
    ps.sort(key=lambda tup: tup[0])
    pss=[]
    sub_set=[]
    for pair in ps:
        if len(sub_set)==0:
            sub_set.append(pair)
        else:
            if adjacent(pair[0],sub_set[-1][0],susp_gap):
                sub_set.append(pair)
            else:
                if len(sub_set)>=susp_size:
                    pss.append(sub_set)
                sub_set=[pair]
    if len(sub_set)>=susp_size:
        pss.append(sub_set)
    psr=[]
    for pss_i in pss:
        pss_i.sort(key=lambda tup: tup[1])
        sub_set=[]
        for pair in pss_i:
            if len(sub_set)==0:
                sub_set.append(pair)
            else:
                if adjacent(pair[1],sub_set[-1][1],src_gap):
                    sub_set.append(pair)
                else:
                    if len(sub_set)>=src_size:
                        psr.append(sub_set)
                    sub_set=[pair]
        if len(sub_set)>=src_size:
            psr.append(sub_set)
    plags=[]
    for psr_i in psr:
        plags.append([(min([x[1] for x in psr_i]),max([x[1] for x in psr_i])),(min([x[0] for x in psr_i]),max([x[0] for x in psr_i]))]) 
    return plags,psr

def remove_overlap3(src_bow,susp_bow,plags):
    plags.sort(key=lambda tup: tup[1][0])
    res=[]
    flag=0
    i=0
    while i<len(plags):
        cont_ol=0
        if flag==0:
            for k in range(i+1,len(plags)):
                if plags[k][1][0]-plags[i][1][1]<=0:
                    cont_ol+=1
        else:
            for k in range(i+1,len(plags)):
                if plags[k][1][0]-res[-1][1][1]<=0:
                    cont_ol+=1
        if cont_ol==0:
            if flag==0:
                res.append(plags[i])
            else:
                flag=0
            i+=1
        else:
            ind_max=i
            higher_sim=0.0
            for j in range(1,cont_ol+1):
                if flag==0:
                    sents_i=range(plags[i][1][0],plags[i][1][1]+1)
                    range_i=range(plags[i][0][0],plags[i][0][1]+1)
                else:
                    sents_i=range(res[-1][1][0],res[-1][1][1]+1)
                    range_i=range(res[-1][0][0],res[-1][0][1]+1)
                sents_j=range(plags[i+j][1][0],plags[i+j][1][1]+1)
                sim_i_ol=0.0
                sim_j_ol=0.0
                sim_i_nol=0.0
                sim_j_nol=0.0
                cont_ol_sents=0
                cont_i_nol_sents=0
                cont_j_nol_sents=0
                for sent in sents_i:
                    sim_max=0.0
                    for k in range_i:
                        sim=cosine_measure(susp_bow[sent],src_bow[k])
                        if sim>sim_max:
                            sim_max=sim
                    if sent in sents_j:
                        sim_i_ol+=sim_max
                        cont_ol_sents+=1
                    else:
                        sim_i_nol+=sim_max
                        cont_i_nol_sents+=1
                range_j=range(plags[i+j][0][0],plags[i+j][0][1]+1)
                for sent in sents_j:
                    sim_max=0.0
                    for k in range_j:
                        sim=cosine_measure(susp_bow[sent],src_bow[k])
                        if sim>sim_max:
                            sim_max=sim
                    if sent in sents_i:
                        sim_j_ol+=sim_max
                    else:
                        sim_j_nol+=sim_max
                        cont_j_nol_sents+=1
                sim_i=sim_i_ol/cont_ol_sents
                if cont_i_nol_sents!=0:
                    sim_i=sim_i+(1-sim_i)*sim_i_nol/float(cont_i_nol_sents)
                sim_j=sim_j_ol/cont_ol_sents
                if cont_j_nol_sents!=0:
                    sim_j=sim_j+(1-sim_j)*sim_j_nol/float(cont_j_nol_sents)
                if sim_i>0.99 and sim_j>0.99:
                    if len(sents_j)>len(sents_i):
                        if sim_j>higher_sim:
                            ind_max=i+j
                            higher_sim=sim_j
                elif sim_j>sim_i:
                    if sim_j>higher_sim:
                        ind_max=i+j
                        higher_sim=sim_j
            if flag==0:
                res.append(plags[ind_max])
            elif ind_max!=i:
                del res[-1]
                res.append(plags[ind_max])
            i=i+cont_ol
            flag=1
    return res

def similarity3(plags,psr,src_bow,susp_bow,src_gap,src_gap_least,susp_gap,susp_gap_least,src_size,susp_size,th3):
    res=[]
    i=0
    range_i=len(plags)
    while i<range_i:
        src_d={}
        for j in range(plags[i][0][0],plags[i][0][1]+1):
            src_d=sum_vect(src_d,src_bow[j])
        susp_d={}
        for j in range(plags[i][1][0],plags[i][1][1]+1):
            susp_d=sum_vect(susp_d,susp_bow[j])
        #if dice_coeff(src_d,susp_d)<=th3:# or cosine_measure(src_d,susp_d)<=0.40:
        if cosine_measure(src_d,susp_d)<=th3:
            if src_gap-src_gap_least>0 and susp_gap-susp_gap_least>0:#Do until substraction +1
                (temp1,temp2)=integrate_cases(psr[i],src_gap-1,susp_gap-1,src_size,susp_size)
                if len(temp1)==0:
                    return []
                res2=similarity3(temp1,temp2,src_bow,susp_bow,src_gap-1,src_gap_least,susp_gap-1,susp_gap_least,src_size,susp_size,th3)
                if len(res2)!=0:
                    res.extend(res2)
            i+=1
        else:
            res.append(plags[i])
            i+=1
    return res

def tokenize(text,voc={},offsets=[],sents=[],rem_sw='no'):
    """
    INPUT:  text: Text to be pre-processed
            voc: vocabulary used in the text with idf
            offsets: start index and length of each sentence
            sents: sentences of the text without tokenization
    OUTPUT: Returns a list of lists representing each sentence divided in tokens
    """
    #text.replace('\0x0', ' ')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents.extend(sent_detector.tokenize(text))
    offsets.extend([(a,b-a) for (a,b) in sent_detector.span_tokenize(text)])
    sent_tokens=[nltk.TreebankWordTokenizer().tokenize(sent) for sent in sents]
    stemmer=nltk.stem.porter.PorterStemmer()
    sent_tokens_pp=[]
    stopwords=flist.flist().words()
    cont=0
    for tokens in sent_tokens:
        if rem_sw=='no':
            temp={}
            for i in [stemmer.stem(word.lower()) for word in tokens if re.match(r'([a-zA-Z]|[0-9])(.)*',word)]:
                if temp.has_key(i):
                    temp[i]+=1
                else:
                    temp[i]=1
        elif rem_sw=='50':
            stopwords=flist.flist().words50()
            temp={}
            for i in [stemmer.stem(word.lower()) for word in tokens if re.match(r'([a-zA-Z]|[0-9])(.)*',word) and word.lower() not in stopwords]:
                if temp.has_key(i):
                    temp[i]+=1
                else:
                    temp[i]=1
        else:
            temp={}
            for i in [stemmer.stem(word.lower()) for word in tokens if re.match(r'([a-zA-Z]|[0-9])(.)*',word) and word.lower() not in stopwords]:
                if temp.has_key(i):
                    temp[i]+=1
                else:
                    temp[i]=1
        if len(temp)>0:
            sent_tokens_pp.append(temp)
            for i in temp.keys():
                if voc.has_key(i):
                    voc[i]+=1
                else:
                    voc[i]=1
            cont=cont+1
        else:
            del offsets[cont]
    return sent_tokens_pp

def serialize_features(susp, src, features, outdir):
    """ Serialze a feature list into a xml file.
    The xml is structured as described in
    http://www.webis.de/research/corpora/pan-pc-12/pan12/readme.txt
    The filename will follow the naming scheme {susp}-{src}.xml and is located
    in the current directory.  Existing files will be overwritten.

    Keyword arguments:
    susp     -- the filename of the suspicious document
    src      -- the filename of the source document
    features -- a list containing feature-tuples of the form
                ((start_pos_susp, end_pos_susp),
                 (start_pos_src, end_pos_src))
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')

    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[1][0]))
        feature.setAttribute('this_length', str(f[1][1] - f[1][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[0][0]))
        feature.setAttribute('source_length', str(f[0][1] - f[0][0]))
        root.appendChild(feature)

    doc.writexml(open(outdir + susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml', 'w'),
                 encoding='utf-8')


# Plagiarism pipeline
# ===================

""" The following class implement a very basic baseline comparison, which
aims at near duplicate plagiarism. It is only intended to show a simple
pipeline your plagiarism detector can follow.
Replace the single steps with your implementation to get started.
"""

class SGSPLAG:
    def __init__(self, susp, src, outdir):
        """ Parameters. """
        self.th1=0.33
        self.th2=0.33
        self.th3=0.40
        self.src_gap=4
        self.src_gap_least=2
        self.susp_gap=4
        self.susp_gap_least=2
        self.src_size=1
        self.susp_size=1
        self.min_sentlen=3
        self.min_plaglen=150
        self.rssent='no'
        self.tf_idf_p='yes'
        self.rem_sw='no'
        
        self.susp = susp
        self.src = src
        self.susp_file = os.path.split(susp)[1]
        self.src_file = os.path.split(src)[1]
        self.susp_id = os.path.splitext(susp)[0]
        self.src_id = os.path.splitext(src)[0]
        self.src_voc={}
        self.susp_voc={}
        self.src_offsets=[]
        self.susp_offsets=[]
        self.src_sents=[]
        self.susp_sents=[]
        self.output = self.susp_id + '-' + self.src_id + '.xml'
        self.detections = None
        self.outdir=outdir

    def process(self):
        """ Process the plagiarism pipeline. """
        #=======================================================================
        # if not os.path.exists(self.output):
        #    os.mkdir(self.output)
        #=======================================================================
        self.preprocess()
        self.detections = self.compare()
        self.postprocess()

    def preprocess(self):
        """ Preprocess the suspicious and source document. """
        src_fp = codecs.open(self.src, 'r', 'utf-8')
        self.src_text = src_fp.read()
        src_fp.close()
        self.src_bow=tokenize(self.src_text,self.src_voc,self.src_offsets,self.src_sents,self.rem_sw)
        ss_treat(self.src_bow,self.src_offsets,self.min_sentlen,self.rssent)
            
        susp_fp = codecs.open(self.susp, 'r', 'utf-8')
        self.susp_text = susp_fp.read()
        susp_fp.close()
        self.susp_bow=tokenize(self.susp_text,self.susp_voc,self.susp_offsets,self.susp_sents,self.rem_sw)
        ss_treat(self.susp_bow,self.susp_offsets,self.min_sentlen,self.rssent)
        
        if self.tf_idf_p=='yes':
            tf_idf(self.src_bow,self.src_voc,self.susp_bow,self.susp_voc)

    def compare(self):
        """ Test a suspicious document for near-duplicate plagiarism with regards to
        a source document and return a feature list.
        """
        ps=[]
        detections=[]
        for c in range(len(self.susp_bow)):
            for r in range(len(self.src_bow)):
                if cosine_measure(self.susp_bow[c],self.src_bow[r])>self.th1 and dice_coeff(self.susp_bow[c],self.src_bow[r])>self.th2:
                    ps.append((c,r))
        (plags,psr)=integrate_cases(ps,self.src_gap,self.susp_gap,self.src_size,self.susp_size)
        (plags2,psr2)=integrate_cases(ps,self.src_gap+20,self.susp_gap+20,self.src_size,self.susp_size)
        #=======================================================================
        # for i in range(len(plags)):
        #     print plags[i]#,psr[i]
        #=======================================================================
        plags=similarity3(plags,psr,self.src_bow,self.susp_bow,self.src_gap,self.src_gap_least,self.susp_gap,self.susp_gap_least,self.src_size,self.susp_size,self.th3)
        plags2=similarity3(plags2,psr2,self.src_bow,self.susp_bow,self.src_gap+20,self.src_gap_least,self.susp_gap+20,self.susp_gap_least,self.src_size,self.susp_size,self.th3)
        #=======================================================================
        # for i in range(len(plags)):
        #     print plags[i]#,psr[i]
        #=======================================================================
        plags=remove_overlap3(self.src_bow,self.susp_bow,plags)
        plags2=remove_overlap3(self.src_bow,self.susp_bow,plags2)
        #=======================================================================
        # for i in range(len(plags)):
        #     print plags[i]#,psr[i]
        #=======================================================================
        sum_src=0
        sum_susp=0
        for plag in plags2:
            arg1=(self.src_offsets[plag[0][0]][0],self.src_offsets[plag[0][1]][0]+self.src_offsets[plag[0][1]][1])
            arg2=(self.susp_offsets[plag[1][0]][0],self.susp_offsets[plag[1][1]][0]+self.susp_offsets[plag[1][1]][1])
            sum_src=sum_src+(arg1[1]-arg1[0]);
            sum_susp=sum_susp+(arg2[1]-arg2[0]);
            
        if sum_src>=3*sum_susp:
            for plag in plags2:
                arg1=(self.src_offsets[plag[0][0]][0],self.src_offsets[plag[0][1]][0]+self.src_offsets[plag[0][1]][1])
                arg2=(self.susp_offsets[plag[1][0]][0],self.susp_offsets[plag[1][1]][0]+self.susp_offsets[plag[1][1]][1])
                if arg1[1]-arg1[0]>=self.min_plaglen and arg2[1]-arg2[0]>=self.min_plaglen: 
                    detections.append([arg1,arg2])
        else:
            for plag in plags:
                arg1=(self.src_offsets[plag[0][0]][0],self.src_offsets[plag[0][1]][0]+self.src_offsets[plag[0][1]][1])
                arg2=(self.susp_offsets[plag[1][0]][0],self.susp_offsets[plag[1][1]][0]+self.susp_offsets[plag[1][1]][1])
                if arg1[1]-arg1[0]>=self.min_plaglen and arg2[1]-arg2[0]>=self.min_plaglen: 
                    detections.append([arg1,arg2])
        return detections

    def postprocess(self):
        """ Postprocess the results. """
        serialize_features(self.susp_file, self.src_file, self.detections, self.outdir)

# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located.
    """
    if len(sys.argv) == 5:
        t1=time.time()
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        outdir = sys.argv[4]
        if outdir[-1] != "/":
            outdir+="/"
        lines = open(sys.argv[1], 'r').readlines()
        for line in lines:
            print(line)
            susp, src = line.split()
            sgsplag_obj = SGSPLAG(os.path.join(suspdir, susp),
                                os.path.join(srcdir, src), outdir)
            sgsplag_obj.process()
        t2=time.time()
        print(t2-t1)
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan13-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {out-dir}"]))