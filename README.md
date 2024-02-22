# Awesome RAG
A curated list of retrieval-augmented generation (RAG) in large language models.


![cc](https://github.com/coree/awesome-rag/assets/5042747/de9c3103-3959-4942-9a52-02156c4bf3a4)



**Table of Content:**
- [Awesome RAG](#awesome-rag)
  - [Papers](#papers)
    - [Survey](#survey)
    - [General](#general)
  - [Resources](#resources)
    - [Lectures, Talks, Tutorials](#lectures-talks-tutorials)
  - [Tools](#tools)
  - [Other Collections](#other-collections)

## Papers

<!-- Paper Template
- **Title**  
  *Auther* (Year) `[Tag]`   
  [`[Paper]`](Link) [`[Code]`](Link) [`[Blog]`](Link) 
-->

<!-- Citation Count Badge
![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F[SEMANTIC_SCHOLAR_PAPER_ID]%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)
-->
### Survey

**2024**

- **Retrieval-Augmented Generation for Large Language Models: A Survey**  
  *Gao et al.* (2024) `[arXiv]`   
  [`[Paper]`](https://arxiv.org/pdf/2312.10997.pdf) [`[Code]`](https://github.com/Tongji-KGLLM/RAG-Survey)


**2023**

- **Benchmarking Large Language Models in Retrieval-Augmented Generation**  
  *Chen et al.* (2023) `[arXiv]`   
  [`[Paper]`](https://arxiv.org/abs/2309.01431) [`[Code]`](Link)


**2022**

- **A Survey on Retrieval-Augmented Text Generation**  
  *Li et al.* (2022) `[arXiv]`   
  [`[Paper]`](https://arxiv.org/abs/2202.01110)
  
### General

**2024**

- **Learning to Retrieve In-Context Examples for Large Language Models**  
  *Wang et al.* (2024) `[EACL]`   
  [`[Paper]`](https://arxiv.org/abs/2307.07164) [`[Code]`](https://github.com/microsoft/LMOps/tree/main/llm_retriever) 


**2023**
- **Active Retrieval Augmented Generation**  
  *Jiang et al.* (2023) `[EMNLP]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2305.06983) [`[Code]`](https://github.com/jzbjyb/FLARE)

- **REPLUG: Retrieval-Augmented Black-Box Language Models**  
  *Shi et al.* (2023) `[arXiv]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2301.12652)

- **Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study**  
  *Wang et al.* (2023) `[EMNLP]`   
  [`[Paper]`](https://arxiv.org/abs/2304.06762) [`[Code]`](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro)

- **InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining**  
  *Wang et al.* (2023) `[arXiv]`   
  [`[Paper]`](https://arxiv.org/abs/2310.07713) [`[Code]`](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro)

- **Retrieve Anything To Augment Large Language Models**  
  *Zhang et al.* (2023) `[arXiv]`   
  [`[Paper]`](https://arxiv.org/abs/2310.07554) [`[Code]`](https://github.com/FlagOpen/FlagEmbedding)  

- **Reimagining Retrieval Augmented Language Models for Answering Queries**  
  *Tan et al.* (2023)  `[ACL]`   
  [`[Paper]`](https://arxiv.org/abs/2306.01061) 

- **In-Context Retrieval-Augmented Language Models**   
  *Ram et al.* (2023)  `[TACL]`  `[Architecture]`   
  [`[Paper]`](https://arxiv.org/abs/2302.00083) [`[Code]`](https://github.com/AI21Labs/in-context-ralm)

- **Query Rewriting for Retrieval-Augmented Large Language Models**  
  *Ma et al.* (2023)  `[EMNLP]`    
  [`[Paper]`](https://arxiv.org/abs/2305.14283) [`[Code]`](https://github.com/xbmxb/RAG-query-rewriting)

- **Pre-computed memory or on-the-fly encoding? A hybrid approach to retrieval augmentation makes the most of your compute**  
  *de Jong et al.* (2023)  `[PMLR]`   
  [`[Paper]`](https://arxiv.org/abs/2301.10448) 

- **Improving language models by retrieving from trillions of tokens**   
  *Borgeaud et al.* (2023)  `[PMLR]` `[Architecture]`        
  [`[Paper]`](https://arxiv.org/abs/2112.04426) [`[Blog]`](https://deepmind.google/discover/blog/improving-language-models-by-retrieving-from-trillions-of-tokens/)

- **Universal Information Extraction with Meta-Pretrained Self-Retrieval**   
  *Cong et al.* (2023) `[ACL]`     
  [`[Paper]`](https://aclanthology.org/2023.findings-acl.251/) [`[Code]`](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/metaretriever)

- **RAVEN: In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models**  
  *Huang et al.* (2023) `[arXiv]`    
    [`[Paper]`](https://arxiv.org/abs/2308.07922)

- **Unlimiformer: Long-Range Transformers with Unlimited Length Input**  
  *Bertsch et al.* (2023) `[NeurIPS]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2305.01625) [`[Code]`](https://github.com/abertsch72/unlimiformer)

- **Nonparametric Masked Language Modeling**  
  *Min et al.* (2023) `[ACL]` `[Training]`      
  [`[Paper]`](https://arxiv.org/abs/2212.01349) [`[Code]`](https://github.com/facebookresearch/NPM)

**2022**

- **Recitation-Augmented Language Models**  
  *Sun et al.* (2022)  `[ICLR]`    
  [`[Paper]`](https://arxiv.org/abs/2210.01296) [`[Code]`](https://github.com/Edward-Sun/RECITE)

- **Atlas: Few-shot Learning with Retrieval Augmented Language Models**  
    *Izacard et al.* (2022) `[Training]` `[JMLR]`    
    [`[Paper]`](https://arxiv.org/abs/2208.03299) [`[Code]`](https://github.com/facebookresearch/atlas) [`[Blog]`](https://research.facebook.com/blog/2023/1/atlas-few-shot-learning-with-retrieval-augmented-language-models/)

- **You can't pick your neighbors, or can you? When and how to rely on retrieval in the kNN-LM**  
  *Drozdov et al.* (2022) `[ACL]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2210.15859)

 - **Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval**  
  *Alon et al.* (2022) `[ICML]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2201.12431) [`[Code]`](https://github.com/neulab/retomaton)

- **Memorizing Transformers**  
  *Wu et al.* (2022)  `[ICLR]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2203.08913) [`[Code]`](https://github.com/princeton-nlp/TRIME)

- **Training Language Models with Memory Augmentation**  
  *Zhong et al.* (2022) `[EMNLP]` `[Training]`      
  [`[Paper]`](https://arxiv.org/abs/2205.12674)

- **Unsupervised Dense Information Retrieval with Contrastive Learning**  
  *Izacard et al.* (2022) `arXiv` `[Training]`      
  [`[Paper]`](https://arxiv.org/abs/2112.09118) [`[Code]`](https://github.com/facebookresearch/contriever)

- **Teaching language models to support answers with verified quotes**  
  *Menick et al.* (2022) `[arXiv]` `[Application]`      
  [`[Paper]`](https://arxiv.org/abs/2203.11147)

- **kNN-Prompt: Nearest Neighbor Zero-Shot Inference**  
  *Shi et al.* (2022) `[EMNLP]` `[Application]`       
  [`[Paper]`](https://arxiv.org/abs/2205.13792) [`[Code]`](https://github.com/swj0419/kNN_prompt)

**2021**
- **Efficient Nearest Neighbor Language Models**  
  *He et al.* (2021) `[EMNLP]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2109.04212) [`[Code]`](https://github.com/jxhe/efficient-knnlm)

- **Mention Memory: incorporating textual knowledge into Transformers through entity mention attention**  
  *de Jong et al.* (2021) `[arXiv]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2110.06176) [`[Code]`](https://github.com/google-research/language/tree/master/language/mentionmemory)


**2020**

- **REALM: Retrieval-Augmented Language Model Pre-Training**  
  *Guu et al.* (2020) `[PMLR]` `[Architecture]`      
  [`[Paper]`](https://arxiv.org/abs/2002.08909) [`[Code]`](https://github.com/google-research/language/blob/master/language/realm/README.md) [`[HuggingFace]`](https://huggingface.co/docs/transformers/model_doc/realm)

- **Generalization through Memorization: Nearest Neighbor Language Models**  
  *Khandelwal et al.* (2020) `[ICLR]` `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/1911.00172) [`[Code]`](https://github.com/urvashik/knnlm)

- **Entities as Experts: Sparse Memory Access with Entity Supervision**  
  *Févry et al.* (2020) `[EMNLP]`  `[Architecture]`       
  [`[Paper]`](https://arxiv.org/abs/2004.07202)

- **Dense Passage Retrieval for Open-Domain Question Answering**  
  *Karpukhin et al.* (2020) `[EMNLP]` `[Training]`    
  [`[Paper]`](https://arxiv.org/abs/2004.04906) [`[Code]`](https://github.com/facebookresearch/DPR)

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
  *Lewis et al.* (2020) `[NeurIPS]`  
  [`[Paper]`](https://arxiv.org/abs/2005.11401) [`[HuggingFace]`](https://huggingface.co/facebook/rag-token-nq) ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F58ed1fbaabe027345f7bb3a6312d41c5aac63e22%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)


## Resources

### Lectures, Talks, Tutorials

**2024**

- **Modular RAG and RAG Flow**  
    *Yunfan Gao* (2024) `[Tutorial]`  
    [`[Blog I]`](https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-%E2%85%B0-e69b32dc13a3)
    [`[Blog II]`](https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-ii-77b62bf8a5d3)


**2023**

- **Stanford CS25: V3 I Retrieval Augmented Language Models**  
  *Douwe Kiela* (2023) `[Lecture]`  
  [`[Video]`](https://www.youtube.com/watch?v=mE7IDf2SmJg&ab_channel=StanfordOnline)

- **Building RAG-based LLM Applications for Production**  
  *Anyscale* (2023) `[Tutorial]`  
  [`[Blog]`](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

- **Multi-Vector Retriever for RAG on tables, text, and images**  
  *LangChain* (2023) `[Tutorial]`  
  [`[Blog]`](https://blog.langchain.dev/semi-structured-multi-modal-rag)

- **Retrieval-based Language Models and Applications**  
  *Asai et al.* (2023) `[Tutorial]`  `[ACL]`  
  [`[Website]`](https://acl2023-retrieval-lm.github.io/) [`[Video]`](https://us06web.zoom.us/rec/play/6fqU9YDLoFtWqpk8w8I7oFrszHKW6JkbPVGgHsdPBxa69ecgCxbmfP33asLU3DJ74q5BXqDGR2ycOTFk.93teqylfi_uiViNK?canPlayFromShare=true&from=share_recording_detail&continueMode=true&componentName=rec-play&originRequestUrl=https%3A%2F%2Fus06web.zoom.us%2Frec%2Fshare%2FNrYheXPtE5zOlbogmdBg653RIu7RBO1uAsYH2CZt_hacD1jOHksRahGlERHc_Ybs.KGX1cRVtJBQtJf0o)

- **Advanced RAG Techniques: an Illustrated Overview**  
  *Ivan Ilin* (2023) `[Tutorial]`   
  [`[Blog]`](https://towardsai.net/p/machine-learning/advanced-rag-techniques-an-illustrated-overview)

  
## Tools

<!-- Tool Template
- **Name**  
  *Description*  
  [`[Website]`](Link)
-->

- **LangChain**  
  *LangChain is a framework for developing applications powered by language models.*  
  [`[Website]`](https://www.langchain.com/)

- **LlamaIndex**  
  *LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models.*  
  [`[Website]`](https://www.llamaindex.ai/)

- **Verba**  
  *Verba is an open-source application designed to offer an end-to-end, streamlined, and user-friendly interface for Retrieval-Augmented Generation (RAG) out of the box.*  
  [`[Website]`](https://verba.weaviate.io/)

- **NEUM**  
  *Open-source RAG framework optimized for large-scale and real-time data.*  
  [`[Website]`](https://www.neum.ai/)

- **Unstructured**  
  *Unstructured.io offers a powerful toolkit that handles the ingestion and data preprocessing step, allowing you to focus on the more exciting downstream steps in your machine learning pipeline. Unstructured has over a dozen data connectors that easily integrate with various data sources, including AWS S3, Discord, Slack, Wikipedia, and more.*  
  [`[Website]`](https://unstructured.io/)

## Other Collections

- [Awesome LLM RAG](https://github.com/jxzhangjhu/Awesome-LLM-RAG)
- [Awesome RAG](https://github.com/frutik/Awesome-RAG)
- [Awesome LLM with RAG](https://github.com/HKUST-AI-Lab/Awesome-LLM-with-RAG)
- [RAG-Survey](https://github.com/Tongji-KGLLM/RAG-Survey)
- [Awesome LLM Reader](https://github.com/HITsz-TMG/awesome-llm-reader): A Repository of Retrieval-augmented LLMs

