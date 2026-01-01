import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
    dimensions=1536,
)

# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings,
# )

vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"),
    embedding=embeddings,
)

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunk of specified size"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i: i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> Dict[str, Any]:
    """Extract documents from a batch of URLs"""
    try: 
        log_info(
            f"TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(
            input={"urls": urls, "extract_depth": "advanced"}
        )
        # Check if response contains an error
        if isinstance(docs, dict) and "error" in docs:
            log_error(f"TavilyExtract: Batch {batch_num} returned error: {docs['error']}")
            return {"results": [], "error": docs["error"]}
        
        log_success(
            f"TavilyExtract: Completed batch {batch_num} - extracted {len(urls)} URLs"
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Error processing batch {batch_num}: {e}")
        return {"results": [], "error": str(e)}

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception: {result}")
            failed_batches += 1
        elif not result:
            # Empty result (from error in extract_batch)
            failed_batches += 1
        else:
            # Handle different response structures
            if isinstance(result, dict) and "results" in result:
                pages = result["results"]
            elif isinstance(result, list):
                pages = result
            else:
                log_warning(f"TavilyExtract: Unexpected result structure: {type(result)}")
                continue
            
            for extracted_page in pages:
                if isinstance(extracted_page, dict) and "raw_content" in extracted_page:
                    document = Document(
                        page_content=extracted_page["raw_content"], 
                        metadata={"source": extracted_page.get("url", "")}
                    )
                    all_pages.append(document)

    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(
            f"TavilyExtract: Failed to extract {failed_batches} batches during extraction."
        )
    return all_pages

async def index_document_async(documents: List[Document], batch_size: int = 50):
    """Processes documents in batches asynchronously"""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"Vectortore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i: i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(
        f"Vectortore Indexing: Created {len(batches)} batches of {batch_size} documents each",        
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Sucessfullt added batch {batch_num}/{len(batches)} {len(batch)} documents"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num}: {e}")
            return False
        return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed succesfully {successful}/{len(batches)}"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )
    
tavily_crawl = TavilyCrawl()
async def main():
    """Main async function to orchestyrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl documentation from http://python.langchain.com",
        Colors.PURPLE,
        )
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(
        f"TavilyMap: Successfully mapped {len(site_map['results'])} URLs from documentation site"
    )
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 1,
        "extract_depth": "advanced",
    })

    # Split urls into batches (smaller batches to avoid Tavily API limits)
    url_batches = chunk_urls(list(site_map['results']), chunk_size=5)
    log_info(
        f"URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE,
    )

    # Extract documents from url
    all_docs = await async_extract(url_batches)

    # Split documents into chunks
    log_header("DOCUMENTS CHUNKING PHASE")
    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents with 4000 chunks size and 200 overlap size",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process Documents Successfully
    await index_document_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETED")
    log_success("Documentation ingestion pipeline completed successfully")
    log_info("Summary: Colors.BOLD")
    log_info(f"     . URLs mapped: {len(site_map['results'])}")
    log_info(f"     . Documents extracted: {len(all_docs)}")
    log_info(f"     . Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())

