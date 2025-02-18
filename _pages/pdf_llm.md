---
layout: default
title: "Applying RAG to PDF summarization"
permalink: /pdf_llm/
---


## A PDF summarizing and querying tool built with RAG and LangChain abstractions.
<!-- The goal here was to create a simple Python program that could automatically parse technical documents, particularly academic papers within the realm of machine/deep learning, and return summaries with a reasonable amount of detail. Inspired by the template laid out by Dr. Vincent Gregoire [here](https://vincent.codes.finance/posts/documents-llm/), I make some modifications:
- updating the summary stuff document chain with the latest iteration of Langchain
- replacing the "map-reduce" approach used with a RAG pipeline, sourcing a database of embeddings of mathematical and machine learning papers/lectures
- augmenting the extraction of terms from the PDFs, to improve LLM parsing -->

The goal of this project was to create a toy demonstration of LLM-based summarization and querying methods. With these tools becoming increasingly commonplace, I wanted to understand what it takes to implement RAG and how effective these models can be, even applied to more niche or technical domains. Settling on a PDF summary tool that I could use for generating quick summaries of research papers posted to arXiv made sense: it would be a good way to build something that I can use, and that is in a sufficiently challenging domain area to push the models.



### A brief overview of RAG and LLM summarization.
Generating summaries of text via LLM is one of the more straightforward and intuitive use cases of large language models. Given their strength in both processing and generating natural language, the idea of passing a large wall of text extracted from a PDF to the model and getting back a condensed summary is natural. Similarly, if a language model is able to summarize a given input of text, then you as a user should be able to query that model about the content within that text. Modern language models are incredibly capable, with extremely large context windows in the hundreds of thousands of tokens: for this project, I used Meta's llama3.1 model that has a context window of 128k, but the largest and most capable models such as Google's Gemini series or OpenAI's popular GPT models have context windows that exceed a million tokens. In theory, this means that given enough computational resources, it could be possible to simply dump the entire content of a document into the model and ask it to return a summary of the input tokens. However, this has a number of problems:
- The larger a context window, the "harder" the model has to work in some sense to generate output tokens -- having to process sucha large amount of input means slower computation and output.
- Language models are known to hallucinate, and can sometimes fail to capture all of the input information. In particular, as input size increases, language models increasingly fail to capture all the contained information and effectively maintain context and relationships. If you think of the attention mechanism underlying transformers and generative language models as [a smoothing operation](http://bactra.org/notebooks/nn-attention-and-transformers.html#attention), then this sort of information loss with larger input sizes is unsurprising. This can also increase the chances of a hallucation by the LLM, and generally renders querying much less efficient.
- Finally, I do not have access to large amounts of compute, 400-billion parameter models, or stacks of GPUs. I want to run this locally on my Mac laptop.

The above considerations result in some important considerations when it comes to summaries and querying.
- For summaries, we make use of the LangChain document stuffing chain to abstract away the document chunking, such that the entire input content can be effectively passed to the LLM as a single prompt.
- For querying, we have two options. A common method for querying a large document or set of documents is called the map-reduce approach, where the input is chunked and mapped via a set of queries into some context set; this context is then used to generate a response to the query in the "reduce" stage. On top of this framework, retrieval-augmented generation (RAG) can be introduced, in which additional context can be retrieved from a vector database and provided to the LLM based on its similarity to the query. This similarity is typically measured by vector cosine similarity, but other metrics can be used. RAG allows for additional information to be efficiently provided to the LLM, which helps prevent hallucination and improve the model response. However, due to the computational cost of calculating, storing, and retrieving the embeddings for the additional context documents, RAG can be slower. 

### Results
The three simple pipelines work as intended, thanks primarily to the convenient abstractions and chains provided by the LangChain library. By leveraging Ollama to serve and query llama3.1 locally, using a quantized 8B parameter install, I was able to efficiently access the model. RAG is evidently more powerful than map-reduce for querying, but for simple summarization, the stuffing chain is surprisingly effective and not too computationally heavy. The summarization does help me screen papers, which is nice. In a real-life use, I would separate this function from the querying/RAG functions in main, which are much more computationally intensive and not always needed.

### TODOs and improvements
A major area of improvement is the processing of the PDF input; a more thorough parsing and cleaning of the content that PyPDFLoader extracts from the documents would likely improve the model response, as information would be cleaner with less noise that is currently still contained in the PDf extraction (such as citations, author names, etc). 

It could also be interesting to improve the RAG/querying functionality by enabling live inference -- a user can query documents from within the Streamlit interface and re-prompt the model. This would require some restructuring of the code and optimization of the pipeline (i.e. efficient persisting of the vector store and further improving the PDF content parsing). 
